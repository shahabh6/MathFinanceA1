import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA

dates = ['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10',
         '2020-01-13', '2020-01-14', '2020-01-15']

def calculate_yields(df):
    yields = []
    df['date'] = df['date'].astype('datetime64[ns]')
    df['MaturityDate'] = df['MaturityDate'].astype('datetime64[ns]')
    for row in df.itertuples():
        num_coupons, price, rate = row.NumCoupons, row.pClose, row.CouponRate
        # dirty_price = (row.DaysSince*rate/365) + price
        yield_rate = npf.irr(cash_flows(num_coupons, price, rate))
        yields.append(yield_rate*2)
    df["Yields"] = pd.Series(yields)
    return df


def plot_yield_curve(df):
    for i in range(len(dates)):
        new_df = df.loc[df['date'] == dates[i]]
        plt.plot(new_df['MaturityDate'], new_df["Yields"], marker = 'o')
    plt.legend(dates)
    plt.xlabel("Maturity Date")
    plt.ylabel("Yield to maturity")
    plt.title("5-Year Yield Curve - CAD Govt. Bonds")
    plt.xticks(rotation=40)
    plt.show()


def cash_flows(num_coupons, price, rate):
    flows = [-1*price]
    payment = rate/2
    for i in range(num_coupons):
        flows.append(payment)
    if len(flows) > 1:
        flows[-1] = 100+payment
    else:
        flows.append(100+payment)
    return flows


def spot_curve(df):
    spot_df = df.copy()
    interpolated_spots = []
    for i in range(len(dates)):
        curr_df = spot_df.loc[spot_df['date'] == dates[i]]
        to_plot_spot = calculate_spot_rates(curr_df)
        plt.plot(to_plot_spot['MaturityDate'], to_plot_spot["SpotRates"], marker = 'o')
        forward_df = to_plot_spot[['MaturityDate', "SpotRates"]]
        forward_df.set_index("MaturityDate", inplace=True)
        # # linear interpolatation to construct spot curve.
        upsampled_df = forward_df.resample("D").interpolate()
        interpolated_spots.append(upsampled_df)
    plt.legend(dates)
    plt.xlabel("Maturity Date")
    plt.ylabel("Spot Rate")
    plt.title("5-Year Spot Curve - CAD Govt. Bonds")
    plt.xticks(rotation=0)
    plt.show()
    forward_rate_list = []

    for i in range(len(dates)):
        x_label = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']
        forward_rates = []
        curr_date = dates[i]
        plus_one = '2021' + curr_date[4:]
        plus_one_rate = interpolated_spots[i].at[plus_one, "SpotRates"]
        forward_rates.append(plus_one_rate)
        for j in range(2, 5):
            plus_j = '202' + str(j) + curr_date[4:]
            plus_j_rate = interpolated_spots[i].at[plus_j, "SpotRates"]
            forward_j_rate = (((1+plus_j_rate)**j)/(1+plus_one_rate))**(1/(j-1)) - 1
            forward_rates.append(forward_j_rate)
        fw_df = pd.DataFrame()
        fw_df['x_axis'] = x_label
        fw_df['fw_rates'] = forward_rates
        forward_rate_list.append(fw_df)
        plt.plot(fw_df['x_axis'], fw_df["fw_rates"], marker = 'o')
    plt.legend(dates)
    plt.xlabel("Period")
    plt.ylabel("Forward Rate")
    plt.title("Forward Curve - CAD Govt. Bonds")
    plt.xticks(rotation=0)
    plt.show()
    return forward_rate_list

def calculate_spot_rates(spot_df):
    spot_rates = []

    # yield_df = spot_df[['MaturityDate', "Yields"]]
    # yield_df.set_index("MaturityDate", inplace=True)
    # # linear interpolatation to construct yield curve.
    # upsampled_df = yield_df.resample("D").interpolate()

    for row in spot_df.itertuples():
        currDate, pClose, rate, numCoupons, yield_rate = row.date, row.pClose, row.CouponRate, \
                                                         row.NumCoupons, row.Yields
        flows = cash_flows(numCoupons, pClose, rate)
        T = [1/4, 3/4, 5/4, 7/4, 9/4, 10/4, 13/4, 14/4, 17/4, 18/4, 19/4, 21/4]
        # print(flows)
        if len(flows) == 2:
            spot_rates.append(-math.log(-flows[0]/flows[1])/T[0])
        elif len(flows) >= 3:
            for k in range(len(flows[1:-1])):
                p_diff = pClose - flows[k+1]*math.exp(-1*spot_rates[k]*(T[k]))
            next_spot_rate = (-math.log(p_diff/flows[-1]))/(T[len(flows)-1])
            #print(flows)
            #print(next_spot_rate)
            spot_rates.append(next_spot_rate)
    spot_df["SpotRates"] = spot_rates

    return spot_df

def yield_covariance(yield_df):
    bond_dates = ['2021-03-01', '2022-03-01', '2023-03-01', '2024-03-01', '2025-03-01']
    log_df = pd.DataFrame()
    # print(yield_df)
    for i in range(len(bond_dates)):
        # print(bond_dates[i])
        restricted_df = yield_df.loc[yield_df['MaturityDate'] == bond_dates[i]]
        # print(restricted_df)
        yields_differenced = np.log(restricted_df["Yields"]).diff().iloc[1:].reset_index(drop=True)
        # print(yields_differenced)
        log_df[bond_dates[i]] = yields_differenced
    # print(log_df.transpose())
    result = pd.DataFrame(np.cov(log_df.transpose()))
    # print(result)
    return result

def forward_covariance(fw_rate_list):
    construct_df = pd.DataFrame()
    for i in range(len(dates)):
        fw_df = fw_rate_list[i]
        construct_df[dates[i]] = fw_df["fw_rates"]
    # print(abs(construct_df))
    fw_diff = np.log(abs(construct_df)).diff(axis=1).drop(columns=['2020-01-02'])
    # print(fw_diff)
    result = pd.DataFrame(np.cov(fw_diff))
    # print(result)
    return result

def eigen(cov_matrix):
    # Return eigenvalues of the covariance matrix.
    print(pd.DataFrame(LA.eigvals(cov_matrix)).transpose())
    # Return eigenvector associated with highest/first eigenvalue.
    print(LA.eig(cov_matrix)[1][0])


if __name__ == "__main__":
    print("hello_world")
    data = pd.read_csv("5yrBonds.csv")
    df = calculate_yields(data)
    plot_yield_curve(df)
    fw_rate_list = spot_curve(df)
    yield_matrix = yield_covariance(df)
    fw_matrix = forward_covariance(fw_rate_list)
    eigen(yield_matrix)
    eigen(fw_matrix)