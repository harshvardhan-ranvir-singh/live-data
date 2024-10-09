import requests
import pandas as pd
import numpy as np
import time
#import xlwings as xw
from bs4 import BeautifulSoup
import datetime
import pytz
import streamlit as st
import csv
import warnings

warnings.simplefilter('ignore')

st.set_page_config(page_title="Dashboard", layout="wide")

TWO_PERCENT_MARKET_PRICE_CE = 0.0
TWO_PERCENT_MARKET_PRICE_PE = 0.0

exchange = "NSE"
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def last_thursdays(year):
    exp = []
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if month == 1 or month == 2 or month == 3 or month == 4 or month == 5 or month == 6 or month == 7 or month == 8 or month == 9:
            date = f"{year}-0{month}-01"
        if month == 10 or month == 11 or month == 12:
            date = f"{year}-{month}-01"

        # we have a datetime series in our dataframe...
        df_Month = pd.to_datetime(date)

        # we can easily get the month's end date:
        df_mEnd = df_Month + pd.tseries.offsets.MonthEnd(1)

        # Thursday is weekday 3, so the offset for given weekday is
        offset = (df_mEnd.weekday() - 3) % 7

        # now to get the date of the last Thursday of the month, subtract it from
        # month end date:
        df_Expiry = df_mEnd - pd.to_timedelta(offset, unit='D')
        exp.append(df_Expiry.date())

    return exp


today_year = datetime.datetime.now().year
exp_date_list = last_thursdays(today_year)
DATE_LIST = []
TODAY = datetime.date.today()
for i in range(len(exp_date_list)):
    x = (exp_date_list[i] - TODAY).days
    if x >= 0:
        DATE_LIST.append(exp_date_list[i].strftime('%d-%m-%Y'))
EXP_OPTION = DATE_LIST[0]


def current_market_price(ticker, exchange):
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"

    for _ in range(1000000):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        class1 = "YMlKec fxKbKc"

        price = float(soup.find(class_=class1).text.strip()[1:].replace(",", ""))
        yield price

        time.sleep(5)

def fifty_two_week_high_low(ticker, exchange):
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    class1 = "P6K39c"

    price = soup.find_all(class_=class1)[2].text
    low_52_week = float(price.split("-")[0].strip()[1:].replace(",", ""))
    high_52_week = float(price.split("-")[1].strip()[1:].replace(",", ""))
    return low_52_week, high_52_week


def get_dataframe(ticker, exp_date_selected):
    while True:
        try:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

            main_url = "https://www.nseindia.com/"
            response = requests.get(main_url, headers=headers)
            cookies = response.cookies

            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
            option_chain_data = requests.get(url, headers=headers, cookies=cookies)

            data = option_chain_data.json()["records"]["data"]
            ocdata = []

            for i in data:
                for j, k in i.items():
                    if j == "CE" or j == "PE":
                        info = k
                        info["instrumentType"] = j
                        ocdata.append(info)

            df = pd.DataFrame(ocdata)



            strikes = df.strikePrice.unique().tolist()
            strike_size = int(strikes[int(len(strikes) / 2) + 1]) - int(strikes[int(len(strikes) / 2)])

            for price in current_market_price(ticker, exchange):
                two_percent_cmp_ce = price + 0.02 * price
                two_percent_cmp_pe = price - 0.02 * price
                TWO_PERCENT_MARKET_PRICE_CE = two_percent_cmp_ce
                TWO_PERCENT_MARKET_PRICE_PE = two_percent_cmp_pe
                break

            print(TWO_PERCENT_MARKET_PRICE_CE, TWO_PERCENT_MARKET_PRICE_PE)

            # access dataframe for atm price
            atm_ce = int(round(TWO_PERCENT_MARKET_PRICE_CE / strike_size, 0) * strike_size)
            print(atm_ce)

            output_ce = pd.DataFrame()

            atm_pe = int(round(TWO_PERCENT_MARKET_PRICE_PE / strike_size, 0) * strike_size)
            output_pe = pd.DataFrame()

            for _ in range(5):

                # (for ce)
                ab = True
                while ab:

                    fd = df[df['strikePrice'] == atm_ce]

                    if fd.empty:
                        print("empty df ce", atm_ce)
                        atm_ce = atm_ce + 0.5
                        if atm_ce > strikes[-1]:
                            break
                    else:
                        ab = False

                # print(fd)

                # (for pe)
                ab_pe = True
                while ab_pe:

                    fd_pe = df[df['strikePrice'] == atm_pe]

                    if fd_pe.empty:
                        print("empty df pe", atm_pe)
                        atm_pe = atm_pe - 0.5
                    else:
                        ab_pe = False

                # print(fd_pe)

                # (for ce)convert expiry date in particular format
                fd = fd.reset_index()
                for i in range(len(fd)):
                    expiry_date_str = fd["expiryDate"].iloc[i]
                    temp_expiry = datetime.datetime.strptime(expiry_date_str, '%d-%b-%Y')
                    result_expiry = temp_expiry.strftime('%d-%m-%Y')
                    fd.at[i, "expiryDate"] = result_expiry
                # print(fd)
                # print(type(fd["expiryDate"].iloc[0]))

                # (for pe) convert expiry date in particular format
                fd_pe = fd_pe.reset_index()
                for i in range(len(fd_pe)):
                    expiry_date_str_pe = fd_pe["expiryDate"].iloc[i]
                    temp_expiry_pe = datetime.datetime.strptime(expiry_date_str_pe, '%d-%b-%Y')
                    result_expiry_pe = temp_expiry_pe.strftime('%d-%m-%Y')
                    fd_pe.at[i, "expiryDate"] = result_expiry_pe

                adjusted_expiry = exp_date_selected
                adjusted_expiry_pe = exp_date_selected

                # (subset_ce (CE))
                subset_ce = fd[(fd.instrumentType == "CE") & (fd.expiryDate == adjusted_expiry)]
                # print(subset_ce)
                output_ce = pd.concat([output_ce, subset_ce])

                # (subset_pe (PE))
                subset_pe = fd_pe[(fd_pe.instrumentType == "PE") & (fd_pe.expiryDate == adjusted_expiry_pe)]
                # print(subset_pe)
                output_pe = pd.concat([output_pe, subset_pe])

                # (for CE)
                atm_ce += strike_size

                # (for PE)
                atm_pe -= strike_size

            output_ce = output_ce[["strikePrice", "expiryDate", "lastPrice", "instrumentType"]]
            output_pe = output_pe[["strikePrice", "expiryDate", "lastPrice", "instrumentType"]]

            output_ce.reset_index(drop=True, inplace=True)
            output_pe.reset_index(drop=True, inplace=True)

            return output_ce, output_pe

        except Exception as e:
            pass


def highlight_ratio(val, column_name):
    if column_name == "CE Premium%":
        color = 'background-color: paleturquoise' if val > 1 else ""
        return color
    if column_name == "CE (Premium+SP)%":
        color = 'background-color: wheat' if val > 5 else ""
        return color
    if column_name == "PE Premium%":
        color = 'background-color: paleturquoise' if val > 1 else ""
        return color
    if column_name == "PE (Premium+SP)%":
        color = 'background-color: wheat' if val > 5 else ""
        return color






@st.fragment
def frag_table(table_number, selected_option='UBL', exp_option=EXP_OPTION):
    st.write("---")
    shares = pd.read_csv("FNO Stocks - All FO Stocks List, Technical Analysis Scanner.csv")
    share_list = list(shares["Symbol"])
    selected_option = selected_option.strip()
    share_list.remove(selected_option)
    share_list = [selected_option] + share_list

    exp_date_list_sel = DATE_LIST.copy()
    print("LIST: ", exp_date_list_sel)

    print("EXP_OPTION:", (exp_option))

    print(type(exp_option))
    print(type(exp_date_list_sel[0]),exp_date_list_sel)


    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### Share List')
        selected_option = st.selectbox(label="", options=share_list, key="share_list" + str(table_number), label_visibility='collapsed')
        lot_size = shares[shares["Symbol"] == selected_option]['Jun-24'].item()
    with c2:
        st.markdown('##### Expiry List')
        exp_option = st.selectbox(label="", options=exp_date_list_sel, key="exp_list" + str(table_number), label_visibility='collapsed')
        if selected_option in share_list:
            ticker = selected_option
            output_ce, output_pe = get_dataframe(ticker, exp_option)
            ########################################## Stock LTP and Matrix #######################################
            stock_ltp = 0.0
            for price in current_market_price(ticker, exchange):
                stock_ltp = price
                break
            low_52_week, high_52_week = fifty_two_week_high_low(ticker, exchange)

        # ********************************** MATRIX ******************************************
        l1, l2 = len(output_ce), len(output_pe)
        if l1 < l2:
            fin_len = l1
        else:
            fin_len = l2
        matrix = np.zeros((fin_len, 4))
        df = pd.DataFrame(matrix, columns=["CE Premium%", "CE (Premium+SP)%", "PE Premium%", "PE (Premium+SP)%"])

        for i in range(len(df)):
            df.at[i, "CE Premium%"] = round((output_ce["lastPrice"].iloc[i] / stock_ltp) * 100, 2)
            df.at[i, "CE (Premium+SP)%"] = round(
                (((output_ce["strikePrice"].iloc[i] - stock_ltp) + output_ce["lastPrice"].iloc[i]) / stock_ltp) * 100,
                2)
            df.at[i, "PE Premium%"] = round((output_pe["lastPrice"].iloc[i] / stock_ltp) * 100, 2)
            df.at[i, "PE (Premium+SP)%"] = round(
                (((stock_ltp - output_pe["strikePrice"].iloc[i]) + output_pe["lastPrice"].iloc[i]) / stock_ltp) * 100,
                2)
        # ************************************************************************************
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        st.markdown('##### CMP:  ' + str(stock_ltp))
    with d2:
        st.markdown('##### Lot Size:  ' + str(lot_size))
    with d3:
        st.markdown('##### Contract Value:  ' + str(lot_size*stock_ltp))
    with d4:
        st.markdown('##### Time:  ' + datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S"))
    with d5:
        st.markdown('##### 52 week low:  ' + str(low_52_week))
    with d6:
        st.markdown('##### 52 week high:  ' + str(high_52_week))



    filters = st.columns(4)
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ls = []
    n=1
    with filters[1]:
        nested_filters = st.columns(2)
        ind = 0
        for column in df.columns.tolist()[:2]:
            with nested_filters[ind]:
                ls.append(st.selectbox(
                                f'Filter {column}',
                                values,
                                key='filter_list' + str(n) + "table" + str(table_number)

                            ))
                n += 1
                ind += 1
    with filters[3]:
        nested_filters = st.columns(2)
        ind = 0
        for column in df.columns.tolist()[2:]:
            with nested_filters[ind]:
                ls.append(st.selectbox(
                            f'Filter {column}',
                            values,
                            key='filter_list' + str(n) + "table" + str(table_number)

                            ))
                n += 1
                ind += 1


    col1, col2, col3, col4 = st.columns(4)
    df_ce = df[['CE Premium%', 'CE (Premium+SP)%']]
    df_pe = df[['PE Premium%', 'PE (Premium+SP)%']]
    df_ce = df_ce[(df_ce['CE Premium%'] >= ls[0]) & (df_ce['CE (Premium+SP)%'] >= ls[1])]
    df_pe = df_pe[(df_pe['PE Premium%'] >= ls[2]) & (df_pe['PE (Premium+SP)%'] >= ls[3])]
    #df = df[(df['CE Premium%'] >= ls[0]) & (df['CE (Premium+SP)%'] >= ls[1]) & (df['PE Premium%'] >= ls[2]) & (df['PE (Premium+SP)%'] >= ls[3])]
    df_ce_index = df_ce.index
    output_ce = output_ce.loc[df_ce_index]
    df_pe_index = df_pe.index
    output_pe = output_pe.loc[df_pe_index]
    with col1:
        output_ce = output_ce.rename(columns={'strikePrice': 'Strike Price',
                                              'expiryDate': 'Expiry Date',
                                              'lastPrice': 'Last Price',
                                              'instrumentType': 'Type'})
        output_ce = output_ce.style.set_properties(**{'text-align': 'center', 'background-color': 'palegreen'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}])
        output_ce = output_ce.format({'Last Price': "{:.2f}".format, 'Strike Price': "{:.1f}".format})
        st.markdown('<style>.col_heading{text-align: center}</style>', unsafe_allow_html=True)
        output_ce.columns = ['<div class="col_heading">'+col+'</div>' for col in output_ce.columns]
        st.write(output_ce.to_html(escape=False), unsafe_allow_html=True)
    with col2:
        # df_ce = df[['CE Premium%', 'CE (Premium+SP)%']]
        df_ce = df_ce.style.applymap(lambda val: highlight_ratio(val, 'CE Premium%'), subset=['CE Premium%'])
        df_ce = df_ce.applymap(lambda val: highlight_ratio(val, 'CE (Premium+SP)%'), subset=['CE (Premium+SP)%'])
        df_ce = df_ce.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}])
        df_ce = df_ce.format({'Last Price': "{:.2f}".format, 'Strike Price': "{:.1f}".format})
        st.markdown('<style>.col_heading{text-align: center}</style>', unsafe_allow_html=True)
        df_ce.columns = ['<div class="col_heading">' + col + '</div>' for col in df_ce.columns]
        st.write(df_ce.to_html(escape=False), unsafe_allow_html=True)
    with col3:
        output_pe = output_pe.rename(columns={'strikePrice': 'Strike Price',
                                              'expiryDate': 'Expiry Date',
                                              'lastPrice': 'Last Price',
                                              'instrumentType': 'Type'})
        output_pe = output_pe.style.set_properties(
            **{'text-align': 'center', 'background-color': 'antiquewhite'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}])
        output_pe = output_pe.format({'Last Price': "{:.2f}".format, 'Strike Price': "{:.1f}".format})
        st.markdown('<style>.col_heading{text-align: center}</style>', unsafe_allow_html=True)
        output_pe.columns = ['<div class="col_heading">' + col + '</div>' for col in output_pe.columns]
        st.write(output_pe.to_html(escape=False), unsafe_allow_html=True)
    with col4:
        # df_pe = df[['PE Premium%', 'PE (Premium+SP)%']]
        df_pe = df_pe.style.applymap(lambda val: highlight_ratio(val, 'PE Premium%'), subset=['PE Premium%'])
        df_pe = df_pe.applymap(lambda val: highlight_ratio(val, 'PE (Premium+SP)%'), subset=['PE (Premium+SP)%'])
        df_pe = df_pe.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}])
        df_pe = df_pe.format({'Last Price': "{:.2f}".format, 'Strike Price': "{:.1f}".format})
        st.markdown('<style>.col_heading{text-align: center}</style>', unsafe_allow_html=True)
        df_pe.columns = ['<div class="col_heading">' + col + '</div>' for col in df_pe.columns]
        st.write(df_pe.to_html(escape=False), unsafe_allow_html=True)



    if ('share_list2' in st.session_state) and ('share_list3' in st.session_state):
        curr = pd.DataFrame({'table1': [st.session_state["share_list1"]],
                             'exp1': [st.session_state["exp_list1"]],
                             'table2': [st.session_state["share_list2"]],
                             'exp2': [st.session_state["exp_list2"]],
                             'table3': [st.session_state["share_list3"]],
                             'exp3': [st.session_state["exp_list3"]],
                             'timestamp': [datetime.datetime.now()]
                             })
        if len(hist_df) > 30:
            curr.to_csv('history.csv', mode='w', index=False, header=True)
        else:
            curr.to_csv('history.csv', mode='a', index=False, header=False)
    st.write("---")
st.markdown('## LIVE OPTION CHAIN ANALYSIS (OPTSTK)')
hist = pd.read_csv("history.csv")
hist_df = pd.DataFrame(hist)

print(len(hist_df))

if len(hist_df) > 1:
    last_rec = hist_df.tail(1)
    print(last_rec)
    frag_table(1, last_rec['table1'].item(), last_rec['exp1'].item())
    frag_table(2, last_rec['table2'].item(), last_rec['exp2'].item())
    frag_table(3, last_rec['table3'].item(), last_rec['exp3'].item())
else:
    frag_table(1, 'RELIANCE')
    frag_table(2, 'VEDL')
    frag_table(3, 'INFY')