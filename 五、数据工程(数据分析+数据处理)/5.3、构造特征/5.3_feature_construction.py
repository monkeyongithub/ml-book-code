# -*- coding: utf-8 -*-
# 文件用途：演示领域经验、纯技术与业务规则三类特征构造。
# 对应章节/小节：第 5 章 5.3 构造特征
# 作者：NYY
# 创建日期：2024-12-18
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.3、构造特征/5.3_feature_construction.py"
# 备注：部分示例使用可选的外部服务或模型，按需开启。

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.dataset import iris_df as load_iris_df
from shared.dataset import titanic_df as load_titanic_df

plt.rcParams['font.sans-serif'] = ['FangSong', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / '五、数据工程(数据分析+数据处理)' / 'output'

# Step1、载入共用的 Titanic 数据；Iris 仅在聚类或树示例启用时加载。
titanic_df, label_name = load_titanic_df()


# Step2、用开关选择领域经验、纯技术与业务规则的特征构造示例。
design_features_by_K_time = False 
design_features_by_K_address = False
design_features_by_K_relu = False
design_features_by_K_cluster = False
design_features_by_T_transformed = False
design_features_by_T_bin = True
design_features_by_T_interactive = False    # 一直 False
design_features_by_T_cross = False          # 一直 False
design_features_by_T_tree = True
design_features_by_T_RFM = True
design_features_by_Business = True

if design_features_by_K_cluster or design_features_by_T_tree:
    iris_df, _ = load_iris_df()


# Step3、按书稿 5.3.1—5.3.3 执行已启用的特征构造分支。
# 5.3.1 基于常识经验或领域知识构造特征
if design_features_by_K_time:
    # T1、构造与时间信息相关的特征(基于时序特征构造)
    def design_features_by_K_time():
        import numpy as np
        import matplotlib.pyplot as plt
        # 模拟销售数据集
        np.random.seed(0)
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sales_data = pd.DataFrame({
            'Date': date_range,
            'Sales': np.random.randint(100, 1000, size=len(date_range))
        })
        print(sales_data)
        # 可视化时间序列
        plt.figure(figsize=(10, 6))
        plt.plot(sales_data['Date'], sales_data['Sales'])
        plt.title('Daily Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True)
        plt.show()
        # 构造时间戳特征
        sales_data['Year'] = sales_data['Date'].dt.year
        sales_data['Month'] = sales_data['Date'].dt.month
        sales_data['Day_of_week'] = sales_data['Date'].dt.dayofweek
        # 构造时间间隔特征
        sales_data['Time_Delta'] = sales_data['Date'].diff().dt.days
        sales_data['Lag_1'] = sales_data['Sales'].shift(1)
        # 构造周期性特征
        sales_data['Quarter'] = sales_data['Date'].dt.quarter
        # 构造滑动窗口特征
        sales_data['Rolling_Mean'] = sales_data['Sales'].rolling(window=7).mean()
        OUTPUT_DIR.mkdir(exist_ok=True)
        sales_data.to_csv(OUTPUT_DIR / 'sales_data.csv', index=False)
        # 输出构造后的数据集
        print(sales_data)
    design_features_by_K_time()


if design_features_by_K_address:
    # T2、从坐标计算可复现的距离特征；地名和海拔需要额外可信数据源。
    def design_features_by_K_address():
        from geopy.distance import geodesic

        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            'latitude': rng.uniform(31.1, 31.4, size=10),
            'longitude': rng.uniform(121.3, 121.6, size=10),
        })
        business_district = (31.2304, 121.4737)
        subway_station = (31.2367, 121.5067)
        data['distance_to_business_km'] = data.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), business_district).km,
            axis=1,
        )
        data['distance_to_subway_km'] = data.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), subway_station).km,
            axis=1,
        )
        print(data)

    design_features_by_K_address()


if design_features_by_K_relu:
    # T3、构造规则衍生特征
    def design_features_by_K_relu(df,name_feature):
        # 从姓名中提取称呼信息
        df['title'] = df[name_feature].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        print(df['title'].value_counts())
        # 基于常识经验，对这些称呼、称谓进行规范化统一
        title_standard = {"Mr": "Mr",
                          "Ms": "Mrs", "Mrs": "Mrs", "Mme": "Mrs", "Dona": "Mrs",
                          "Miss": "Miss", "Mlle": "Miss",
                          "Master": "Master",
                          "Don": "Royalty", "Lady": "Royalty", "Sir": "Royalty", "the Countess": "Royalty", "Jonkheer": "Royalty",
                          "Rev": "Officer", "Dr": "Officer", "Major": "Officer", "Col": "Officer", "Capt": "Officer"}
        df['title_standard'] = df['title'].map(title_standard)
        print(df['title_standard'].value_counts())
        print(df)
    name_feature = 'name'
    design_features_by_K_relu(titanic_df,name_feature)


if design_features_by_K_cluster:
    # T4、基于经验聚类构造特征
    # 利用聚类算法自动构建特征
    def design_features_by_K_cluster():
        # 提取特征
        X = iris_df.drop('target', axis=1)
        # 使用K均值聚类算法对花瓣长度和花瓣宽度进行聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        iris_df['petal_cluster'] = kmeans.fit_predict(X[['petallength', 'petalwidth']])
        # 输出构造的新特征
        print(iris_df[['petallength', 'petalwidth', 'petal_cluster']])
    design_features_by_K_cluster()


# 5.3.2 基于纯技术构造特征
if design_features_by_T_transformed:
    # T1、基于非线性变换构造特征
    def design_features_by_T_transformed(data_name,df,num_features):
        # 对于Box-Cox变换，它要求数据必须严格为正值
        df = df[(df[num_features] > 0).all(axis=1)].copy()
        feature = num_features[0]
        # 对数值型特征进行非线性变换
        import numpy as np
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer
        # 使用幂函数变换进行非线性变换
        df[f'{feature}_log'] = np.log1p(df[feature])
        # 使用Box-Cox变换进行非线性变换
        pt = PowerTransformer(method='box-cox')
        df[f'{feature}_boxcox'] = pt.fit_transform(df[feature].values.reshape(-1, 1))
        # 使用分位数变换进行非线性变换
        qt = QuantileTransformer(output_distribution='normal')
        df[f'{feature}_quantile'] = qt.fit_transform(df[feature].values.reshape(-1, 1))
        # 查看原始数据分布
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 4, 1)
        df[f'{feature}'].hist(bins=20)
        plt.title(f'%s的原始分布(%s数据集)'%(feature,data_name))
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.grid(False)  # 去掉图像中的网格
        # 查看变换后的数据分布
        plt.subplot(1, 4, 2)
        df[f'{feature}_log'].hist(bins=20)
        plt.title(f'执行Log转换后的分布')
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.subplot(1, 4, 3)
        df[f'{feature}_boxcox'].hist(bins=20)
        plt.title(f'执行BoxCox转换后的分布')
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.subplot(1, 4, 4)
        df[f'{feature}_quantile'].hist(bins=20)
        plt.title(f'执行分位数转换后的分布')
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    # 选择数值型特征
    num_features = ['age']
    design_features_by_T_transformed(data_name,titanic_df,num_features)
    num_features = ['fare']
    design_features_by_T_transformed(data_name,titanic_df,num_features)


if design_features_by_T_bin:
    # T2、构造分箱特征：
    def design_features_by_T_bin(titanic_df,num_features):
        label_name = 'target'
        # Step1、选取待分箱的数值列，并填补缺失值。
        titanic_df = titanic_df[[num_features[0],num_features[1],label_name]].copy()
        titanic_df[num_features] = titanic_df[num_features].fillna(titanic_df[num_features].mean())
        print(titanic_df.info())
        # Step2、依次生成统计、聚类和业务规则分箱特征。
        # (1) 从技术角度构造特征
        # 1)、基于统计的方法：等距分箱、等频分箱、等间距分位数、基于标准差
        # 等距分箱：采用cut实现
        titanic_df['%s_equal_width'%num_features[0]] = pd.cut(titanic_df[num_features[0]], bins=3, labels=['低', '中', '高'])
        # 等频分箱
        titanic_df['%s_equal_freq'%num_features[0]] = pd.qcut(titanic_df[num_features[0]], q=3, duplicates='drop')
        # 等间距分位数
        import mapclassify as mc
        quant_bin_counts = mc.Quantiles(titanic_df[num_features[0]],k=5)
        print(quant_bin_counts)
        titanic_df['%s_quant'%num_features[0]] = quant_bin_counts.yb
        # 基于标准差：尽可能地减少组内差异和增加组间差异
        std_bin_counts = mc.StdMean(titanic_df[num_features[0]])
        print(std_bin_counts)
        titanic_df['%s_std'%num_features[0]] = std_bin_counts.yb
        # 2)、基于数据分布的方法：基于自然群聚趋势、基于重尾分布
        nb_bin_counts = mc.NaturalBreaks(titanic_df[num_features[0]],k=5)
        print(nb_bin_counts)
        titanic_df['%s_nb'%num_features[0]] = nb_bin_counts.yb
        htb_bin_counts = mc.HeadTailBreaks(titanic_df[num_features[0]])
        print(htb_bin_counts)
        titanic_df['%s_htb'%num_features[0]] = htb_bin_counts.yb
        # 3)、基于模型算法预测的方法：利用KMeans无监督聚类的分箱
        # 基于无监督聚类的分箱
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        titanic_df['%s_cluster'%num_features[0]] = kmeans.fit_predict(titanic_df[[num_features[0]]])
        # (2) 从业务角度构造特征
        # 基于业务规则/先验知识法
        import numpy as np
        if num_features[0] == 'age':
            know_bins = pd.cut(
                titanic_df['age'], bins=[0, 6, 17, 40, 65, np.inf],
                labels=['儿童', '少年', '中年', '老年', '高龄'],
            )
            titanic_df['age_business_bins'] = know_bins

        # (3) 从是否有监督角度分类构造特征
        # 无监督分箱
        # 采用KBinsDiscretizer实现等距分箱
        from sklearn.preprocessing import KBinsDiscretizer
        k_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        titanic_df['%s_uns_equal_width'%num_features[0]] = k_bins.fit_transform(titanic_df[[num_features[0]]])
        OUTPUT_DIR.mkdir(exist_ok=True)
        titanic_df.to_csv(OUTPUT_DIR / f'titanic_{num_features[0]}_bins.csv', index=False)
        print(titanic_df)

        # Step3、对照原始分布与各方案的分箱结果。
        # 绘制分布曲线图
        plt.figure(figsize=(12, 8))
        # 使用for循环绘制分布曲线图
        bin_methods = [
            num_features[0],
            f'{num_features[0]}_equal_width', f'{num_features[0]}_equal_freq',
            f'{num_features[0]}_quant', f'{num_features[0]}_std',
            f'{num_features[0]}_nb', f'{num_features[0]}_htb',
            f'{num_features[0]}_cluster',
        ]
        if num_features[0] == 'age':
            bin_methods.append('age_business_bins')
        for i, method in enumerate(bin_methods):
            plt.subplot(3, 3, i + 1)
            if method == num_features[0]:  # 如果是画布中的第一张图，则使用密度图可视化
                sns.histplot(x=method, data=titanic_df, kde=True)
            else:  # 否则使用计数图
                sns.countplot(x=method, data=titanic_df)
            plt.xticks(rotation=20)
        plt.tight_layout(pad=1)  # 增加子图之间的间距
        plt.show()
    data_name = 'titanic'
    num_features = ['age', 'fare']
    design_features_by_T_bin(titanic_df,num_features)
    num_features = ['fare', 'age']
    design_features_by_T_bin(titanic_df,num_features)

if design_features_by_T_interactive:
    # T3、用同一组明确的输入演示四则运算、平方和多项式交互。
    def design_features_by_T_interactive():
        from sklearn.preprocessing import PolynomialFeatures

        people = pd.DataFrame({
            'height_cm': [170, 165, 180, 175, 185],
            'weight_kg': [60, 55, 70, 65, 75],
        })
        people['height_m'] = people['height_cm'] / 100
        people['bmi'] = people['weight_kg'] / people['height_m'] ** 2
        people['body_size'] = people['height_cm'] * people['weight_kg']
        poly = PolynomialFeatures(degree=2, include_bias=False)
        matrix = poly.fit_transform(people[['height_cm', 'weight_kg']])
        polynomial = pd.DataFrame(matrix, columns=poly.get_feature_names_out())
        print(people)
        print(polynomial)

        sales = pd.Series([100, 120, 110, 135], name='sales')
        print('销售额二阶差分：', sales.diff(periods=2))

    design_features_by_T_interactive()


if design_features_by_T_cross:
    # T4、交叉类别列并汇总同组数值，展示装箱统计的含义。
    def design_features_by_T_cross():
        cities = pd.DataFrame({
            'coastal': ['是', '是', '否', '否'],
            'capital': ['是', '否', '是', '否'],
            'age_group': ['青年', '青年', '成年', '成年'],
            'population': [120, 80, 100, 60],
        })
        cities['coastal_capital'] = cities['coastal'] + '_' + cities['capital']
        grouped = cities.groupby('age_group')['population'].agg(['sum', 'mean', 'max', 'min'])
        print(cities)
        print(grouped)

    design_features_by_T_cross()


if design_features_by_T_tree:
    # T5、基于树叶节点编码构造特征
    def design_features_by_T_tree(data_name,df,label_name):
        from sklearn.tree import DecisionTreeClassifier, export_graphviz
        import graphviz
        from graphviz.backend import ExecutableNotFound
        df = df.copy()
        # Step1、将类别值编码并填补缺失值，再分离特征和标签。
        for column in df.select_dtypes(include=['category']).columns:
            df[column] = df[column].cat.codes
        df = df.fillna(df.median(numeric_only=True))
        print(df.info())
        X = df.drop(columns=[label_name])  # 特征
        y = df[label_name]  # 标签
        # Step2、训练浅层决策树，用叶节点编号构造新特征。
        # 构建决策树模型
        max_d = 3  # 可比较 4、5 或 None 的树深度
        tree_model = DecisionTreeClassifier(max_depth=max_d)
        tree_model.fit(X, y)
        # 获取叶节点路径和叶节点编码
        leaf_paths = tree_model.decision_path(X)
        leaf_codes = tree_model.apply(X)
        # 统计叶节点编码及其个数
        leaf_codes_series = pd.Series(leaf_codes)
        value_counts_result = leaf_codes_series.value_counts()
        # 输出样本的叶节点编码
        print("样本的叶节点编码：", len(value_counts_result),'\n',value_counts_result)
        # 构造特征：将叶节点编码作为新的特征
        data_with_leaf_codes = df.assign(Leaf_Code=leaf_codes)
        print(data_with_leaf_codes.head())
        # Step3、将树图导出到本章 output 目录，便于对照叶节点编码。
        # 可视化树结构
        dot_data = export_graphviz(tree_model, out_file=None, feature_names=X.columns)
        graph = graphviz.Source(dot_data)
        # 将决策树图保存为PNG格式图片、PDF文件等
        OUTPUT_DIR.mkdir(exist_ok=True)
        try:
            graph.render(
                filename=f'{data_name}_tree_{max_d}', directory=str(OUTPUT_DIR),
                format='png', cleanup=True,
            )
        except ExecutableNotFound:
            print('未安装 Graphviz 系统程序，已跳过树图导出；叶节点编码仍可观察。')
    data_name = 'titanic'
    select_cols = ['sex','age','fare','sibsp','target']
    titanic_tree_df = titanic_df[select_cols].copy()
    design_features_by_T_tree(data_name,titanic_tree_df,label_name)
    data_name = 'iris'
    design_features_by_T_tree(data_name,iris_df,label_name)


if design_features_by_T_RFM:
    # T6、基于统计聚类的RFM方法构造
    def design_features_by_T_RFM():
        # 模拟数据集
        data = {
            'CustomerID': [1, 2, 3, 4, 5],
            'Recency': [20, 10, 5, 30, 15],     # 最近一次购买时间间隔
            'Frequency': [5, 2, 8, 1, 4],       # 购买次数
            'Monetary': [100, 50, 200, 30, 80]  # 购买金额
        }
        # 转换为 DataFrame
        df = pd.DataFrame(data)
        # Step1、保留客户最近购买、购买频次和消费金额三列。
        # Step2、先标准化三个量纲不同的指标，再按相似性聚类。
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
        df_scaled = pd.DataFrame(df_scaled, columns=['Recency', 'Frequency', 'Monetary'])

        # 聚类分析
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)
        # 加入聚类结果到原始数据集中
        df['Cluster'] = df['Cluster'].astype('category')
        print(df)

        # Step3、绘制三维分布，观察新增的客户群组特征。
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制散点图
        colors = ['r', 'g', 'b']  # 不同聚类的颜色
        for cluster, color in zip(range(3), colors):
            ax.scatter(df[df['Cluster'] == cluster]['Recency'],
                       df[df['Cluster'] == cluster]['Frequency'],
                       df[df['Cluster'] == cluster]['Monetary'],
                       c=color,
                       label='Cluster {}'.format(cluster))
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        # 添加图例
        ax.legend()
        # 显示图形
        plt.title('基于RFM方法构造特征')
        plt.show()
    design_features_by_T_RFM()


# T7、图结构特征见本节 5.3.2_graph_features.py。


# 5.3.3 基于业务规则和意义构造特征
if design_features_by_Business:
    def design_features_by_Business():
        import pandas as pd
        # 示例数据集
        data = {
            '借款人行业': ['房地产', '医疗保健', '互联网金融', '房地产', '制造业'],
            '担保人职业': ['自由职业者', '公务员', '大型企业职员', '短期合同工', '私营企业老板']
        }
        # 将行业和职业映射到风险等级
        risk_level_mapping = {
            '借款人行业': {
                '房地产': '高',
                '医疗保健': '低',
                '互联网金融': '中',
                '教育': '低',
                '制造业': '中'
            },
            '担保人职业': {
                '公务员': '低',
                '自由职业者': '高',
                '大型企业职员': '低',
                '短期合同工': '高',
                '私营企业老板': '中'
            }
        }

        # 根据风险等级映射创建新特征
        def construct_risk_level_feature(row):
            borrower_risk_level = risk_level_mapping['借款人行业'][row['借款人行业']]
            guarantor_risk_level = risk_level_mapping['担保人职业'][row['担保人职业']]

            if borrower_risk_level == '高' and guarantor_risk_level == '高':
                return '高'
            elif borrower_risk_level == '低' and guarantor_risk_level == '低':
                return '低'
            else:
                return '中'
        # 创建数据框
        df = pd.DataFrame(data)
        # 构造客户征信风险等级特征
        df['客户征信风险等级'] = df.apply(construct_risk_level_feature, axis=1)
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / 'design_features_by_Business.csv', index=False)
        print(df)
    design_features_by_Business()


# 5.3.4 的深度特征提取示例见 5.3.4_deep_feature_extraction.py。

