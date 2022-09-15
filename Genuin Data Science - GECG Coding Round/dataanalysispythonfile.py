import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ecomData = pd.read_csv('ecom-elasticity-data1.csv', header=None)
ecomData.columns = ["ASIN", "Category", "Date", "Price", "Units Sold", "Month", "Year", "Season", "Revenue"]
# ecomData = np.array(ecomData)

diaperData = ecomData.loc[ecomData['Category'] == 'Diapers']
headphoneData = ecomData.loc[ecomData['Category'] == 'Headphones']
cerealData = ecomData.loc[ecomData['Category'] == 'Cereals']

# monthData = ecomData["Month"].unique()
# diaperPriceData = []
# diaperDemandData = []
# diaperRevenueData = []
# headphonePriceData = []
# headphoneDemandData = []
# headphoneRevenueData = []
# cerealPriceData = []
# cerealDemandData = []
# cerealRevenueData = []
# cnt = 0
# for i in monthData:
#     diaperPriceData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Diapers')]['Price'].mean())
#     diaperDemandData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Diapers')]['Units Sold'].mean())
#     diaperRevenueData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Diapers')]['Revenue'].mean())
#     headphonePriceData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Headphones')]['Price'].mean())
#     headphoneDemandData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Headphones')]['Units Sold'].mean())
#     headphoneRevenueData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Headphones')]['Revenue'].mean())
#     cerealPriceData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Cereals')]['Price'].mean())
#     cerealDemandData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Cereals')]['Units Sold'].mean())
#     cerealRevenueData.append(ecomData.loc[(ecomData['Month'] == i) & (ecomData['Category'] == 'Cereals')]['Revenue'].mean())
#     cnt = cnt + 1

# plt.bar(monthData, diaperRevenueData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
# plt.plot(monthData, diaperRevenueData, marker='x')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Revenue")
# plt.title("Diapers Revenue X Month")
# plt.show()
#
# plt.bar(monthData, headphoneRevenueData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='green')
# plt.plot(monthData, headphoneRevenueData, marker='x', color='green')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Revenue")
# plt.title("Headphones Revenue X Month")
# plt.show()
#
# plt.bar(monthData, cerealRevenueData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='red')
# plt.plot(monthData, cerealRevenueData, marker='x', color='red')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Revenue")
# plt.title("Cereals Revenue X Month")
# plt.show()

# plt.scatter(monthData, diaperPriceData, s = np.array(diaperDemandData)*100, alpha=0.5)
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Monthly Trends in Diapers Category")
# plt.show()
#
# plt.scatter(monthData, headphonePriceData, s = np.array(headphoneDemandData)*100, alpha=0.5)
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Monthly Trends in Headphones Category")
# plt.show()
#
# plt.scatter(monthData, cerealPriceData, s = np.array(cerealDemandData)*100, alpha=0.5)
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Monthly Trends in Cereals Category")
# plt.show()

# plt.plot(monthData, diaperPriceData, label = "Price", marker='x')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Diapers Price X Month")
# plt.show()
#
# plt.plot(monthData, diaperDemandData, label = "Demand", marker='x', color = 'blue')
# plt.bar(monthData, diaperDemandData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Units Sold")
# plt.title("Diapers Demand X Month")
# plt.show()
#
# plt.plot(monthData, headphonePriceData, label = "Price", marker='x', color = 'red')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Headphones Price X Month")
# plt.show()
#
# plt.plot(monthData, headphoneDemandData, label = "Demand", marker='x', color = 'red')
# plt.bar(monthData, headphoneDemandData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='red')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Units Sold")
# plt.title("Headphones Demand X Month")
# plt.show()
#
# plt.plot(monthData, cerealPriceData, label = "Price", marker='x', color = 'green')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Price")
# plt.title("Cereal Price X Month")
# plt.show()
#
# plt.plot(monthData, cerealDemandData, label = "Demand", marker='x', color = 'green')
# plt.bar(monthData, cerealDemandData, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='green')
# plt.xticks(rotation = 90)
# plt.xlabel("Month")
# plt.ylabel("Units Sold")
# plt.title("Cereals Demand X Month")
# plt.show()

# plot1 = plt.subplot2grid((2, 2), (0, 0))
# plot2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
# plot3 = plt.subplot2grid((2, 2), (1, 0))
#
# plot1.scatter(diaperDemandData, diaperPriceData, color = "green")
# plot1.grid()
# plot1.set_title('Diapers')
#
# plot2.scatter(headphoneDemandData, headphonePriceData, color = "red")
# plot2.grid()
# plot2.set_title('Headphones')
#
# plot3.scatter(cerealDemandData, cerealPriceData, color = "blue")
# plot3.grid()
# plot3.set_title('Cereals')
#
# plt.suptitle('Price X Demand', fontsize=16)
# plt.tight_layout()
# plt.show()

totaldiaperdemand = ecomData.loc[(ecomData['Category'] == 'Diapers')]['Units Sold'].sum()
totalheadphonedemand = ecomData.loc[(ecomData['Category'] == 'Headphones')]['Units Sold'].sum()
totalcerealdemand = ecomData.loc[(ecomData['Category'] == 'Cereals')]['Units Sold'].sum()

totaldemand = np.array([totaldiaperdemand, totalheadphonedemand, totalcerealdemand])

mylabels = ["Diapers", "Headphones", "Cereals"]
# plt.pie(totaldemand, labels = mylabels)
# plt.title('Total')
# plt.show()


diaperdemand17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Diapers')]['Units Sold'].sum()
headphonedemand17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Headphones')]['Units Sold'].sum()
cerealdemand17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Cereals')]['Units Sold'].sum()
totaldemand17 = ecomData.loc[(ecomData['Year'] == 2017)]['Units Sold'].sum()

diaperdemand18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Diapers')]['Units Sold'].sum()
headphonedemand18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Headphones')]['Units Sold'].sum()
cerealdemand18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Cereals')]['Units Sold'].sum()
totaldemand18 = ecomData.loc[(ecomData['Year'] == 2018)]['Units Sold'].sum()

diaperdemand19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Diapers')]['Units Sold'].sum()
headphonedemand19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Headphones')]['Units Sold'].sum()
cerealdemand19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Cereals')]['Units Sold'].sum()
totaldemand19 = ecomData.loc[(ecomData['Year'] == 2019)]['Units Sold'].sum()

plotdata = pd.DataFrame({

    "Diapers":[int(diaperdemand17), int(diaperdemand18), int(diaperdemand19)],

    "Headphones":[int(headphonedemand17), int(headphonedemand18), int(headphonedemand19)],

    "Cereals":[int(cerealdemand17), int(cerealdemand18), int(cerealdemand19)]},

    index=["2017", "2018", "2019"])

# plotdata.plot(kind="bar")
# plt.title("Demand X Year")
# plt.xlabel("Year")
# plt.ylabel("Units Sold")
# plt.show()

# seasonData = ecomData["Season"].unique()
# diaperPriceData = []
# diaperDemandData = []
# headphonePriceData = []
# headphoneDemandData = []
# cerealPriceData = []
# cerealDemandData = []
# cnt = 0
# for i in seasonData:
#     diaperPriceData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Diapers')]['Price'].mean())
#     diaperDemandData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Diapers')]['Units Sold'].mean())
#     headphonePriceData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Headphones')]['Price'].mean())
#     headphoneDemandData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Headphones')]['Units Sold'].mean())
#     cerealPriceData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Cereals')]['Price'].mean())
#     cerealDemandData.append(ecomData.loc[(ecomData['Season'] == i) & (ecomData['Category'] == 'Cereals')]['Units Sold'].mean())
#     cnt = cnt + 1
#
# plt.scatter(seasonData, diaperPriceData, s = np.array(diaperDemandData)*100, alpha=0.5)
# plt.xlabel("Season")
# plt.ylabel("Price")
# plt.title("Seasonal Trends in Diapers Category")
# plt.show()
#
# plt.scatter(seasonData, headphonePriceData, s = np.array(headphoneDemandData)*100, alpha=0.5)
# plt.xlabel("Season")
# plt.ylabel("Price")
# plt.title("Seasonal Trends in Headphones Category")
# plt.show()
#
# plt.scatter(seasonData, cerealPriceData, s = np.array(cerealDemandData)*100, alpha=0.5)
# plt.xlabel("Season")
# plt.ylabel("Price")
# plt.title("Seasonal Trends in Cereals Category")
# plt.show()

diapers_price = ecomData.loc[(ecomData['Category'] == "Diapers")]['Price']
diapers_demand = ecomData.loc[(ecomData['Category'] == "Diapers")]['Units Sold']
diapers_sd = np.std(diapers_price.tolist())
diapers_mean = ecomData.loc[(ecomData['Category'] == "Diapers")]['Price'].mean()
diapers_coeff_v = diapers_sd / diapers_mean
diapers_covari = np.cov(diapers_price, diapers_demand)
diapers_corr, _ = pearsonr(diapers_price, diapers_demand)
diapers_great_avg = ecomData.loc[(ecomData['Category'] == "Diapers") & (ecomData['Price'] > diapers_mean)]['Price'].count()
diapers_prob = diapers_great_avg / len(diapers_price)

headphone_price = ecomData.loc[(ecomData['Category'] == "Headphones")]['Price']
headphone_demand = ecomData.loc[(ecomData['Category'] == "Headphones")]['Units Sold']
headphones_sd = np.std(headphone_price.tolist())
headphones_mean = ecomData.loc[(ecomData['Category'] == "Headphones")]['Price'].mean()
headphones_cv = headphones_sd / headphones_mean
headphones_covari = np.cov(headphone_price, headphone_demand)
headphones_corr, _ = pearsonr(headphone_price, headphone_demand)
headphones_great_avg = ecomData.loc[(ecomData['Category'] == "Headphones") & (ecomData['Price'] > headphones_mean)]['Price'].count()
headphones_prob = headphones_great_avg / len(headphone_price)

cereal_price = ecomData.loc[(ecomData['Category'] == "Cereals")]['Price']
cereal_demand = ecomData.loc[(ecomData['Category'] == "Cereals")]['Units Sold']
cereals_sd = np.std(cereal_price.tolist())
cereals_mean = ecomData.loc[(ecomData['Category'] == "Cereals")]['Price'].mean()
cereals_cv = cereals_sd / cereals_mean
cereals_covari = np.cov(cereal_price, cereal_demand)
cereals_corr, _ = pearsonr(cereal_price, cereal_demand)
cereals_great_avg = ecomData.loc[(ecomData['Category'] == "Cereals") & (ecomData['Price'] > diapers_mean)]['Price'].count()
cereals_prob = cereals_great_avg / len(cereal_price)

# diaperrevenue17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Diapers')]['Revenue'].sum()
# headphonerevenue17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Headphones')]['Revenue'].sum()
# cerealrevenue17 = ecomData.loc[(ecomData['Year'] == 2017) & (ecomData['Category'] == 'Cereals')]['Revenue'].sum()
#
# diaperrevenue18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Diapers')]['Revenue'].sum()
# headphonerevenue18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Headphones')]['Revenue'].sum()
# cerealrevenue18 = ecomData.loc[(ecomData['Year'] == 2018) & (ecomData['Category'] == 'Cereals')]['Revenue'].sum()
#
# diaperrevenue19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Diapers')]['Revenue'].sum()
# headphonerevenue19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Headphones')]['Revenue'].sum()
# cerealrevenue19 = ecomData.loc[(ecomData['Year'] == 2019) & (ecomData['Category'] == 'Cereals')]['Revenue'].sum()
#
# plotdata = pd.DataFrame({
#
#     "Diapers":[int(diaperrevenue17), int(diaperrevenue18), int(diaperdemand19)],
#
#     "Headphones":[int(headphonerevenue17), int(headphonerevenue18), int(headphonedemand19)],
#
#     "Cereals":[int(cerealrevenue17), int(cerealrevenue18), int(cerealrevenue19)]},
#
#     index=["2017", "2018", "2019"])
#
# plotdata.plot(kind="bar")
# plt.title("Revenue X Year")
# plt.xlabel("Year")
# plt.ylabel("Revenue")
# plt.show()

# totaldiaperrevenue = ecomData.loc[(ecomData['Category'] == 'Diapers')]['Revenue'].sum()
# totalheadphonerevenue = ecomData.loc[(ecomData['Category'] == 'Headphones')]['Revenue'].sum()
# totalcerealrevenue = ecomData.loc[(ecomData['Category'] == 'Cereals')]['Revenue'].sum()
#
# totalrevenue = np.array([totaldiaperrevenue, totalheadphonerevenue, totalcerealrevenue])
#
# revenuelabels = ["Diapers", "Headphones", "Cereals"]
# plt.pie(totalrevenue, labels = mylabels)
# plt.title('Total Revenue')
# plt.show()

# plt.boxplot(diapers_price, showmeans=True)
# plt.show()
# plt.boxplot(headphone_price, showmeans=True)
# plt.show()
# plt.boxplot(cereal_price, showmeans=True)
# plt.show()

