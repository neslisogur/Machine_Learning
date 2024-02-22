import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('machine_learning/TelcoChurn/Telco-Customer-Churn.csv')
df.head()

df.shape
#(7043, 21)
df.describe().T
df.columns
df.isnull().sum()
df.info()

def grab_col_names(dataframe, cat_th = 10, car_th = 20 ):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisindedir.

    """

    #-----------------------------------KATEGORİK DEĞİŞKENLERİN AYARLANMASI-------------------------------------
    #tipi kategori yada bool olanları seç
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    #integer yada float olup 10dan küçü değerdekileri yakala. Bu veriler numerik ama kategorik kaydedilir.
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    #tipleri kategorik ya da object olup eşsiz sınıf sayısı 20den fazla olanları getirç
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #-----------------------------------NUMERİK DEĞİŞKENLERİN AYARLANMASI-------------------------------------
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float]"]]
    num_cols = [col for col in df.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

#object görünüyor bunu numeric değişkene çeviriyoruz
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
df['Churn'] = df['Churn'].map({"No": 0, "Yes": 1})


df.head()
def cat_summary(dataFrame, col_name, plot = False):
    print(pd.DataFrame({col_name: dataFrame[col_name].value_counts(),
                        "Ratio": 100 * dataFrame[col_name].value_counts() / len(dataFrame)}))
    print('###########################')
    if plot:
        sns.countplot(x=dataFrame[col_name], data=dataFrame)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)


df.head()
#yes  = 1 / no = 0
df.shape
#(7043, 21)
df.describe().T
#sayısal değişkenler: SENIORCITIZEN,TENURE, MONTHLYCHARGES
df.columns
#eksik değer yok
df.isnull().sum()
df.info()

#kategorik değişken ile hedef değişken analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Churn": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


#aykırı gözlem
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#aykırı değer analizi: yok
check_outlier(df, "tenure")
check_outlier(df, "MonthlyCharges")
check_outlier(df, "TotalCharges")


#eksik değer analizi: var (Total Charge)
df.isnull().values.any() #True
df.isnull().sum() #TotalCharge
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    #eksik değer oranı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    #bunları df'e çevir
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

#eksik değere sahip kolon ve eksik değer sayısını getir.
missing_values_table(df)
#eksik değerleri  ortalama ile doldurma
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean()).isnull().sum()
#eksik değer problemi çözüldü
missing_values_table(df) # []

df.head(50)
df["SeniorCitizen"].head()
df[df["SeniorCitizen"] == 1].head()#yaşlı kullanıcılar
df[df["SeniorCitizen"] == 0].head()#genç

df.loc[df["SeniorCitizen"] == 1, 'NEW_AGE'] = 'old'
df.loc[df["SeniorCitizen"] == 0, 'NEW_AGE'] = 'young'
df.groupby("NEW_AGE")["Churn"].mean()
df.loc[df["Partner"] == "Yes", 'NEW_PARTNER'] = 'NOT-ALONE'
df.loc[df["Partner"] == "No", 'NEW_PARTNER'] = 'ALONE'
df.groupby("NEW_PARTNER")["Churn"].mean()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)

df.head()
df[binary_cols].head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]



df = one_hot_encoder(df, ohe_cols)
df.head()


ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)
X.head()
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)