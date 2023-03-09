############################################
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri)
############################################

# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentlerin Oluşturulması
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması

##################################################
# 1. Veri Hazırlama
##################################################
# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#print yapılırken tüm sütunları (değişkenleri) görmek istediğim için:
pd.set_option('display.max_columns', None)

#print tapılırken tüm satırları görmek istediğim için:
# pd.set_option('display.max_rows', None)

#Float değerlerin ondalık kısmını sıfırdan sonra kaç basamak istersem seçmek için:
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel('CRM Analytics/online_retail_II.xlsx', sheet_name='Year 2009-2010')
df = df_.copy()
df.head()

#Faturada iptal olan başında C olanları liste dışına attım.
df = df[~df['Invoice'].str.contains('C', na=False)]

df.describe().T

#Eksi değerlerden kurtulalım Quantity
df = df[df['Quantity'] > 0]

#ID sini bilmediğim musteri için çalışamam bu sebeple eksik verileri kaldıralım:
df.dropna(inplace=True)

df['TotalPrice'] = df['Quantity'] * df['Price']

#Total Transaction her bir faturadaki eşsiz sayı, yani müiterinin işlem sayısı
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})
#Her bir müşterinin eşsiz işlemde(transaction), eşsiz üründe(unit) yaptığı toplam satın(price) alma
#CRM Analitiğinde RFM analizinde yer alan Frequency=Total Transaction, Monetary=TotalPrice
cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################
cltv_c['average_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
cltv_c.head()

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################
#.shape bize satır ve sutunu gösterir. Satırlarda bize toplam müşteri sayısını verir. O. indeksi(4314) seçelim:
cltv_c.shape
cltv_c['purchase_frequency'] = cltv_c['total_transaction'] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################
#Total transaction yapılan işlem sayısını verir. Ben buraya >1 şeklinde filtre yapmalıyım:
cltv_c[cltv_c['total_transaction'] > 1]

#Ben bu değerleri değil işlem yapan sayıyı istiyor. shape metoudyla bunu da bulalım:
repeat_rate = cltv_c[cltv_c['total_transaction'] > 1].shape[0] / cltv_c.shape[0]

#Repeat Rate: en az iki defa alışveriş yapanların tüm müşterilere oranı. Churn(Terk) ise bunun 1 den farkı
churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################
#0.10 şirketlere göre değişen sabit değerdir.
cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
cltv_c.sort_values('cltv', ascending=False).head()

# 18102 ID li müşteri toplam 89 işlemde 124216 farklı ürün alıp 349164.35000 birim ödeme yaptı.
# Average Order Value: 3923.19494

##################################################
# 8. Segmentlerin Oluşturulması
##################################################
#Life Time Value değeri ile bizim için önemli müşterilerin sıralamasını biliyorum:
cltv_c.sort_values('cltv', ascending=False).head()

#Bunları 4 gruba ayıralım
cltv_c['segment'] = pd.qcut(cltv_c['cltv'], 4, ['D', 'C', 'B', 'A'])
cltv_c

#Peki bu yaptığımız mantıklı mı? Kontrol edelim,
cltv_c.groupby('segment').agg(['count', 'mean', 'sum'])

#A segmentinin diğerlerinden açık ara farkı görülüyor B ve C segmentleri bir birine yakın 3 segmentte yapabilirdik.


##################################################
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması
##################################################

def create_cltv_c(dataframe, profit=0.10):

    # Veriyi hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])
    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']
    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()

clv = create_cltv_c(df)
