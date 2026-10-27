# -*- coding: utf-8 -*-
# 文件用途：演示数据对齐、缺失值、异常值和特殊值的清洗方法。
# 对应章节/小节：第 5 章 5.1 数据清洗
# 作者：NYY
# 创建日期：2024-12-17
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.1、数据清洗/5.1_data_cleaning.py"
# 备注：各方法为可单独阅读的示例，可按需开启相应开关。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.dataset import person_info_df, titanic_df as load_titanic_df

plt.rcParams['font.sans-serif'] = ['FangSong', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Step1、用开关选择要观察的清洗示例；数据对齐使用内置小样本。
data_alignment = True           
missing_values_statistics = False
missing_values_handle = False
find_outliers_by_3std = False
find_outliers_by_scatter = False
find_outliers_by_boxplot = False
find_outliers_by_probplot = False
find_outliers_by_LoF = False
find_outliers_by_IF = False
find_outliers_by_DBSCAN = False
del_outliers_by_boxplot = False
del_outliers_by_DT = False
transform_outliers_by_DT = False
detect_special_values = False

# Step2、仅为启用的缺失值、特殊值或异常值分支准备对应数据。
if missing_values_statistics or detect_special_values:
    titanic_df, label_name = load_titanic_df()
if any((
    find_outliers_by_3std, find_outliers_by_scatter, find_outliers_by_boxplot,
    find_outliers_by_probplot, find_outliers_by_LoF, find_outliers_by_IF,
    find_outliers_by_DBSCAN, del_outliers_by_boxplot, del_outliers_by_DT,
    transform_outliers_by_DT,
)):
    person_info_data_df = person_info_df()

# Step3、进入已启用的分支；以下按书稿 5.1.1—5.1.4 分组。
# 5.1.1 数据对齐
if data_alignment:
    # Step1、构造混用数值类型、日期格式与类别名称的微型表。
    data = {
        'age': [18, 20, '32', '44'],
        'date': ['2022年11月', '2022-12', '202206', '202202'],
        'city': ['上海市', '上海', '沪', '北京市'],
        'name': ['Don', 'Lady', 'Sir', 'Jonkheer'],
    }
    df = pd.DataFrame(data)
    # Step2、逐列统一年龄类型、日期格式及类别语义。
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    date_mapping = {
        '2022年11月': '2022-11', '2022-12': '2022-12',
        '202206': '2022-06', '202202': '2022-02',
    }
    df['date'] = df['date'].replace(date_mapping)
    df['city'] = df['city'].replace({'上海': '上海市', '沪': '上海市'})
    df['name'] = df['name'].replace({
        'Don': 'Royalty', 'Lady': 'Royalty',
        'Sir': 'Royalty', 'Jonkheer': 'Royalty',
    })
    # Step3、查看各列对齐后的取值。
    print(df)

if missing_values_statistics:
    # 5.1.2、缺失值的分析与处理
    # 5.1.2.2（1）缺失值的统计与可视化
    # T1、统计法分析检测法
    def missing_values_statistics(df):
        # 统计每列缺失值数量
        missing_values_count = df.isnull().sum()
        # 计算每列缺失值占比
        missing_values_percentage = (missing_values_count / len(df)) * 100
        # 创建缺失值统计表
        missing_values_table = pd.concat([missing_values_count, missing_values_percentage], axis=1)
        # 重命名列名
        missing_values_table.columns = ['缺失值数量', '缺失值占比']
        # 根据缺失值数量降序排序
        missing_values_table = missing_values_table[missing_values_table.iloc[:, 1] != 0].sort_values('缺失值数量',ascending=False)
        return missing_values_table
    titanic_miss_df = missing_values_statistics(titanic_df)
    print("缺失值统计信息：")
    print(titanic_miss_df)

    # T2、可视化检测法
    def missing_values_visualization(df):
        import matplotlib.pyplot as plt
        # 提供的数据
        features = df.index
        missing_values_count = df['缺失值数量'].tolist()
        missing_values_percentage = df['缺失值占比'].tolist()
        print(df.index)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        color = 'tab:blue'
        ax1.bar(features, missing_values_count, color=color)
        ax1.set_xlabel('特征')
        ax1.set_ylabel('缺失值数量', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.plot(features, missing_values_percentage, color=color, marker='o', linestyle='-')
        ax2.set_ylabel('缺失值占比(%)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('不同特征的缺失值数量和缺失值占比(%s数据集)'%data_name)
        plt.tight_layout()
        plt.show()
        print("\n缺失值可视化：")

    data_name = 'titanic'
    missing_values_visualization(titanic_miss_df)


# 5.1.2.2（2）缺失值的处理
if missing_values_handle:
    demo = pd.DataFrame({
        'age': [25, None, 35, 40],
        'income': [100, 200, None, 400],
        'city': ['上海', None, '北京', '上海'],
    })
    # T1、删除法：删除含缺失值的行，可能同时损失其它有效信息。
    print('删除缺失行：', demo.dropna())

    # T2、增补法：数值列用均值或中位数，类别列用众数。
    mean_filled = demo.copy()
    mean_filled['income'] = mean_filled['income'].fillna(mean_filled['income'].mean())
    median_filled = demo.copy()
    median_filled['age'] = median_filled['age'].fillna(median_filled['age'].median())
    mode_filled = demo.copy()
    mode_filled['city'] = mode_filled['city'].fillna(mode_filled['city'].mode().iloc[0])
    print('均值填充：', mean_filled)
    print('中位数填充：', median_filled)
    print('众数填充：', mode_filled)

    # T2、增补法也可按顺序前向填充或线性插值；需先确认顺序有业务意义。
    print('前向填充：', demo['age'].ffill())
    print('线性插值：', demo['income'].interpolate(method='linear'))


# 5.1.3 异常值的分析与处理
# 5.1.3.2（1）异常值检测与分析


data_name = 'person_info'
if find_outliers_by_3std:
    # T1、基于统计的方法：3sigma准则法、散点图、箱线图、QQ分位数图寻找异常点
    def find_outliers_by_3std(df,features):
        # 计算年龄和身高的平均值和标准差
        mean_f01 = df[features[0]].mean()
        std_f01 = df[features[0]].std()
        mean_f02 = df[features[1]].mean()
        std_f02 = df[features[1]].std()

        # 使用3sigma准则检测异常值
        threshold_f01 = 3 * std_f01
        threshold_f02 = 3 * std_f02
        outliers_f01 = df[np.abs(df[features[0]] - mean_f01) > threshold_f01]
        outliers_f02 = df[np.abs(df[features[1]] - mean_f02) > threshold_f02]
        print("异常值(%s)：\n"%features[0] , outliers_f01)
        print("异常值(%s)：\n"%features[1], outliers_f02)

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(df[features[0]], df[features[1]], alpha=0.2, label='Data')
        plt.scatter(outliers_f01[features[0]], outliers_f01[features[1]],alpha=0.5, marker ='*',label='Outliers(%s)'%features[0])
        plt.scatter(outliers_f02[features[0]], outliers_f02[features[1]], alpha=0.5, marker ='<',label='Outliers(%s)'%features[1])
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('使用3-sigma规则检测异常值的散点图(%s数据集)'%data_name)
        plt.legend()
        plt.show()
    features = ['Age','Height']
    find_outliers_by_3std(person_info_data_df,features)


if find_outliers_by_scatter:
    def find_outliers_by_scatter(df,features):
            fig, axes = plt.subplots(figsize=(8, 6))
            # 绘制散点图：可视化2个【数值型】特征分布
            axes.scatter(df[features[0]], df[features[1]])
            axes.set_title('散点图：%s和%s(%s数据集)'%(features[0], features[1],data_name))
            axes.set_xlabel(features[0])
            axes.set_ylabel(features[1])
            plt.show()
    features = ['Age','Height']
    find_outliers_by_scatter(person_info_data_df,features)


if find_outliers_by_boxplot:
    def find_outliers_by_boxplot(df,features):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # 绘制原始箱线图
            df[[features[0], features[1]]].boxplot(ax=axes[0])
            axes[0].set_title('箱线图(原始)：%s和%s'%(features[0], features[1]))
            # 最小-最大缩放数据
            df_scaled = (df - df.min()) / (df.max() - df.min())
            # 绘制归一化后的箱线图
            df_scaled[[features[0], features[1]]].boxplot(ax=axes[1])
            axes[1].set_title('箱线图(归一化后)：%s和%s(%s数据集)'%(features[0], features[1],data_name))
            plt.show()
    features = ['Age','Height']
    find_outliers_by_boxplot(person_info_data_df,features)


if find_outliers_by_probplot:
    def find_outliers_by_probplot(df,features):
        import scipy.stats as stats
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # 绘制QQ分位数图
        stats.probplot(df[features[0]], dist="norm", plot=axes[0])
        axes[0].set_title('Q-Q分位数图：%s'% features[0])
        stats.probplot(df[features[1]], dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q分位数图：%s(%s数据集)'% (features[1],data_name))
        plt.show()
    features = ['Age','Height']
    find_outliers_by_probplot(person_info_data_df,features)


if find_outliers_by_LoF:
    # T2、LOF 比较样本与邻域的局部密度，识别异常点。
    # 使用 LOF 方法检测据集中异常值
    def find_outliers_by_LoF(df,features):
        # 构建并训练 LOF 模型
        from sklearn.neighbors import LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
        outliers = clf.fit_predict(df)
        plt.figure(figsize=(8, 6))
        # 绘制散点图：异常点被标记为颜色较浅的点，而正常点被标记为颜色较深的点
        plt.scatter(df[features[0]], df[features[1]], c=outliers, cmap='coolwarm')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('散点图：利用LOF法检测异常值(%s数据集)'% (data_name))
        plt.colorbar(label='Outlier Factor')
        plt.show()
    features = ['Age','Height']
    find_outliers_by_LoF(person_info_data_df,features)


# T2、孤立森林通过随机划分隔离异常点；与同节 LOF 的原理不同。
if find_outliers_by_IF:
    def find_outliers_by_IF(df,feature):
        from sklearn.ensemble import IsolationForest
        df = df[[feature]]
        # 构建并训练孤立森林模型
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(df)
        # 预测异常值
        outliers = clf.predict(df)
        print(outliers)
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        plt.hist(df[feature][outliers == -1], bins=1, color='red', label='Outliers')
        plt.hist(df[feature][outliers == 1], bins=20, color='blue',label='Inliers')
        plt.title('直方图：利用IF法检测异常值(%s数据集)'% (data_name))
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.legend()
        plt.show()
    features = ['Age','Height']
    for feature in features:
        find_outliers_by_IF(person_info_data_df,feature)


if find_outliers_by_DBSCAN:
    # T3、DBSCAN 将密度聚类中的噪声点视为异常候选。
    # 使用DBSCAN算法检测异常值
    def find_outliers_by_DBSCAN(df,features):
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=10, min_samples=3) # eps用于确定领域的半径参数, 确定核心点的最小样本数参数
        outliers = dbscan.fit_predict(df)
        plt.figure(figsize=(8, 6))
        plt.scatter(df[features[0]], df[features[1]], c=outliers)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('散点图：利用DBSCAN法检测异常值(%s数据集)'% (data_name))
        plt.colorbar(label='Cluster Label')
        plt.show()
    features = ['Age','Height']
    find_outliers_by_DBSCAN(person_info_data_df,features)


# 5.1.3.2（2）异常值处理
if del_outliers_by_boxplot:
    # T2、损失该样本的其它特征信息—直接删除法，基于3σ准则删除异常样本点+箱线图对比可视化
    def del_outliers_by_boxplot(df,feature):
        # 计算特征的平均值和标准差
        mean_age = df[feature].mean()
        std_age = df[feature].std()
        # 使用3sigma准则检测异常值
        threshold_age = 3 * std_age
        # 删除异常样本点
        df_filtered = df[np.abs(df[feature] - mean_age) <= threshold_age]
        # 绘制箱线图比较前后特征分布
        fig, axes = plt.subplots(1, 2, figsize=(8, 7))
        # 原始数据箱线图
        df.boxplot(column=[feature], ax=axes[0])
        axes[0].set_title('箱线图：删除异常值之前')
        # 删除异常值后的箱线图
        df_filtered.boxplot(column=[feature], ax=axes[1])
        axes[1].set_title('箱线图：删除异常值之后(%s数据集)'% (data_name))
        plt.tight_layout()
        plt.show()
    features = ['Age','Height']
    for feature in features:
        del_outliers_by_boxplot(person_info_data_df,feature)


if del_outliers_by_DT:
    # T3.1、数值截断(Data Truncation)
    def del_outliers_by_DT(df, feature, cutoff):
        # 对字段进行截断
        df_truncated = df.copy()
        df_truncated[feature] = np.clip(df_truncated[feature], *cutoff)
        # 绘制带有趋势线的分布柱状图对比前后删除异常值后年龄的分布情况
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True, color='blue')
        plt.title('采用数值截断技术之前的%s分布'%(feature))
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.subplot(1, 2, 2)
        sns.histplot(df_truncated[feature], kde=True, color='red')
        plt.title('采用数值截断技术之后的%s分布(%s数据集)'%(feature,data_name))
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.tight_layout()
        plt.show()
    # 设置异常值的截断范围
    Age_cutoff = (0, 120)  # 年龄截断范围，假设最大年龄为120岁
    Height_cutoff = (100, 220)  # 身高截断范围，假设最小身高为100cm，最大身高为220cm

    for feature, cutoff in [('Age', Age_cutoff), ('Height', Height_cutoff)]:
        del_outliers_by_DT(person_info_data_df, feature, cutoff)


if transform_outliers_by_DT:
    # T3.2、数据变换可以减轻异常值
    def transform_outliers_by_DT(df,feature):
        # 定义数据变换函数（这里以对数变换为例）
        def log_transform(x):
            return np.log(x)
        # 对年龄字段进行对数变换
        df_transformed = df.copy()
        df_transformed['%s_transform'%feature] = log_transform(df_transformed[feature])
        print(df_transformed)
        # 绘制带有趋势线的分布柱状图对比前后删除异常值后年龄的分布情况
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df_transformed[feature], kde=True, color='blue')
        plt.title('采用Log转换之前的%s分布'%(feature))
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.subplot(1, 2, 2)
        sns.histplot(df_transformed['%s_transform'%feature], kde=True, color='red')
        plt.title('采用Log转换之后的%s分布(%s数据集)'%(feature,data_name))
        plt.xlabel('Log(%s)'%feature)
        plt.ylabel('频率')
        plt.tight_layout()
        plt.show()
    features = ['Age','Height']
    for feature in features:
        transform_outliers_by_DT(person_info_data_df,feature)


# 5.1.4 特殊值的分析与处理
if detect_special_values:
    def detect_special_values(data_name,df,features):
        # Step1、去除两个待统计字段中的缺失行。
        df = df.dropna(subset=features)
        # Step2、统计 sibsp 与 parch 各取值的占比。
        sibsp_counts = df[features[0]].value_counts(normalize=True) * 100
        sibsp_counts = sibsp_counts.sort_index()
        # 统计parch字段子类别数量及占比
        parch_counts = df[features[1]].value_counts(normalize=True) * 100
        parch_counts = parch_counts.sort_index()
        # Step3、并排绘图，观察两个字段中特定取值的占比。
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        # 绘制第1个字段柱状图并添加数值标签
        for idx, value in sibsp_counts.items():
            axes[0].bar(idx, value, color='skyblue')
            axes[0].text(idx, value, f'{value:.2f}%', ha='center', va='bottom')
        axes[0].set_title('%s的子类别占比(%s数据集)'%(features[0],data_name))
        axes[0].set_xlabel(features[0])
        axes[0].set_ylabel('占比(%)')
        # 绘制第2个字段柱状图
        for idx, value in parch_counts.items():
            axes[1].bar(idx, value, color='lightgreen')
            axes[1].text(idx, value, f'{value:.2f}%', ha='center', va='bottom')
        axes[1].set_title('%s的子类别占比(%s数据集)'%(features[1],data_name))
        axes[1].set_xlabel(features[1])
        axes[1].set_ylabel('占比(%)')
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    features = ['sibsp', 'parch']
    detect_special_values(data_name,titanic_df,features)


