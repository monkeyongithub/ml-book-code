# -*- coding: utf-8 -*-
# 文件用途：演示目标变量、类别型及数值型特征的分布与关联分析。
# 对应章节/小节：第 5 章 5.2 数据分析与处理
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py"
# 备注：各图示按书稿小节分组，可按需开启。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.dataset import (
    boston_df as load_boston_df,
    iris_df as load_iris_df,
    titanic_df as load_titanic_df,
)

plt.rcParams['font.sans-serif'] = ['FangSong', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Step1、载入各分支共用的 Titanic 数据。
titanic_df, label_name = load_titanic_df()
# Step2、用开关选择图示；仅在对应示例启用时加载 Boston 或 Iris。
Distribution_PK_Train_Test = False 
num_label_distribution = False
cat_label_distribution = False
cat_feature_distribution = False
cat_feature_relation = False
label_cat_features_distribution = False
nums_skew_kurt_plot = False
nums_distribution = False
nums_QQ_distribution = False
num_transformed_distribution_PK = False
num2_features = False
num_features_heatmap = True
nums_label_relation = True
C_PCP_scaler_plot = True
R_PCP_scaler_plot = True
distributio_histplot_mix_features = True
distributio_jointplot_mix_features = True
distributio_jointplot_all_features = True

if num_label_distribution or num2_features or R_PCP_scaler_plot:
    boston_df, _ = load_boston_df()
if cat_label_distribution:
    iris_df, _ = load_iris_df()

# Step3、按书稿 5.2.1—5.2.5 执行已启用的分析分支。
if Distribution_PK_Train_Test:
    # 5.2、数据分析与处理(侧重挖掘信息)
    # 5.2.1、校验两份数据集是否同分布
    def Distribution_PK_Train_Test(df,n_features,c_features):
        # 划分数据集
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        # 绘制数值型特征的直方图和核密度估计图
        plt.figure(figsize=(10, 5))
        for i, feature in enumerate(n_features):
            plt.subplot(1, len(n_features), i+1)
            sns.histplot(train_df[feature], color='blue', kde=True, edgecolor=None, label='训练集', stat='density')
            sns.histplot(test_df[feature], color='red', kde=True, edgecolor=None, label='测试集', stat='density')
            plt.title(f"%s在训练集/测试集中的分布(%s数据集)" % (feature,data_name))
            plt.ylabel('密度')
            plt.legend()
        plt.tight_layout()
        plt.show()
        # 绘制类别型特征的条形图
        plt.figure(figsize=(12, 5))
        for i, feature in enumerate(c_features):
            plt.subplot(1, len(c_features), i+1)
            sns.countplot(x=feature, data=train_df, color='blue', alpha=0.7, label='训练集')
            sns.countplot(x=feature, data=test_df, color='red', alpha=0.7, label='测试集')
            plt.title(f"%s在训练集/测试集中的分布" % (feature))
            plt.ylabel('密度')
            plt.legend()
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    num_features = ['age', 'fare']
    cat_features = ['sex', 'pclass', 'embarked']
    Distribution_PK_Train_Test(titanic_df,num_features,cat_features)


if num_label_distribution:
    # 5.2.2、目标变量的分析与处理
    # A1、回归任务：折线图/QQ分位数图/密度图分布可视化
    def num_label_distribution(data_df_raw,data_name,label_name):
        # 设置图形的风格
        # 创建画布
        plt.figure(figsize=(14, 5))
        # 绘制折线图
        plt.subplot(1, 3, 1)
        sns.lineplot(x=data_df_raw.index, y=label_name, data=data_df_raw)
        plt.title('折线图: %s分布可视化(%s数据集)' % (label_name,data_name))
        plt.xlabel('样本索引')
        plt.ylabel('目标值')
        # 绘制 QQ 分位数图
        from scipy import stats
        plt.subplot(1, 3, 2)
        stats.probplot(data_df_raw[label_name], dist="norm", plot=plt)
        plt.title('QQ图: %s分布可视化(%s数据集)' % (label_name,data_name))
        plt.xlabel('理论分位数')
        plt.ylabel('有序值')
        # 绘制密度图
        plt.subplot(1, 3, 3)
        sns.kdeplot(data_df_raw[label_name], fill=True)
        plt.title('密度图: %s分布可视化(%s数据集)' % (label_name,data_name))
        plt.xlabel('目标值')
        plt.ylabel('密度')
        plt.tight_layout()
        plt.show()
    data_name = 'boston'
    num_label_distribution(boston_df,data_name,label_name)


# 5.2.2 目标变量的分析与处理：分类目标
if cat_label_distribution:
    # A2、分类任务：柱状图/饼状图/词云图分布可视化
    def cat_label_distribution(data_df,data_name,label_name):
        # 创建画布
        plt.figure(figsize=(14, 6))
        # 绘制柱状图
        plt.subplot(1, 3, 1)
        sns.countplot(x='target', data=data_df)
        for index, value in enumerate(data_df['target'].value_counts(ascending=True)):
            plt.text(index, value + 0.5, str(value), ha='center', va='bottom')
        plt.title('计数柱状图：%s分布(%s数据集)' % (label_name,data_name))
        plt.xlabel('类别')
        plt.ylabel('计数')
        # 绘制饼状图
        plt.subplot(1, 3, 2)
        data_df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('饼状图：%s分布(%s数据集)' % (label_name,data_name))
        plt.ylabel('')
        # 生成词云图
        from wordcloud import WordCloud
        plt.subplot(1, 3, 3)
        species_text = ' '.join(data_df['target'])
        wordcloud = WordCloud(width=800, height=400, stopwords=['iris','Iris'], background_color ='white').generate(species_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('词云图：%s分布(%s数据集)' % (label_name,data_name))
        plt.tight_layout()
        plt.show()
    data_name = 'iris'
    cat_label_distribution(iris_df,data_name,label_name)


if cat_feature_distribution:
    # 5.2.3、【类别型】特征分析
    # 5.2.3.1、分布性/独立性分析：
    def cat_feature_distribution(data_df,data_name,cat_features):
        # 创建画布
        plt.figure(figsize=(14, 6))
        # 绘制柱状图
        plt.subplot(1, 3, 1)
        sns.countplot(x=cat_features[0], data=data_df)
        for index, value in enumerate(data_df[cat_features[0]].value_counts(ascending=True)):
            plt.text(index, value + 0.5, str(value), ha='center', va='bottom')
        plt.title('计数柱状图：%s分布(%s数据集)' % (cat_features[0],data_name))
        plt.xlabel(cat_features[0])
        plt.ylabel('计数')
        # 绘制饼状图
        plt.subplot(1, 3, 2)
        data_df[cat_features[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('饼状图：%s分布(%s数据集)' % (cat_features[0],data_name))
        plt.ylabel('')
        # 生成词云图
        from wordcloud import WordCloud
        plt.subplot(1, 3, 3)
        species_text = ' '.join(data_df[cat_features[0]])
        wordcloud = WordCloud(width=800, height=400, stopwords=['iris','Iris'], background_color ='white').generate(species_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('词云图：%s分布(%s数据集)' % (cat_features[0],data_name))
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    cat_features = ['sex','embarked']
    cat_feature_distribution(titanic_df,data_name,cat_features)


if cat_feature_relation:
    # 5.2.3.2、相关性/关联性分析
    # (1)、单特征与label之间的关系可视化
    def cat_feature_relation(data_df, data_name, cat_features, label_name):
        # 创建一个包含两个子图的画布
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        # 绘制第一个子图：性别（sex）与目标变量（survived）的柱状图
        sns.countplot(x=cat_features[0], hue=label_name, data=data_df, ax=axes[0])
        axes[0].set_title('%s与%s的关系分布(%s数据集)' % (cat_features[0], label_name, data_name))
        axes[0].set_ylabel('计数')
        axes[0].set_xlabel(cat_features[0])
        axes[0].legend()
        # 绘制第二个子图：船舱等级（pclass）与目标变量（survived）的柱状图
        sns.countplot(x=cat_features[1], hue=label_name, data=data_df, ax=axes[1])
        axes[1].set_title('%s与%s的关系分布(%s数据集)' % (cat_features[1], label_name, data_name))
        axes[1].set_ylabel('计数')
        axes[1].set_xlabel(cat_features[1])
        axes[1].legend()
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    cat_features = ['sex', 'pclass']
    cat_feature_relation(titanic_df, data_name, cat_features, label_name)


# 5.2.3.2 类别型特征与目标变量的关联分析
if label_cat_features_distribution:
    # (2)、多个组合特征与label之间的关系可视化
    def label_cat_features_distribution(data_df,label_cat_features):
        g = sns.FacetGrid(data_df, col=label_cat_features[1],row=label_cat_features[2])
        # 注意FacetGrid 中，默认情况下只有最底部一排的子图会显示 x 轴标签。
        g.map(sns.countplot,label_cat_features[0],
              palette='Set2',
              )
        g.add_legend()
        plt.show()
    label_cat_features = ['target','pclass','sex']
    label_cat_features_distribution(titanic_df, label_cat_features)


if nums_skew_kurt_plot:
    # 5.2.4、【数值型】特征分析与处理
    # 5.2.4.1、分布性的分析与处理(独立性分析)
    # (1)、统计并可视化所有【数值型】特征的偏态和峰态
    def nums_skew_kurt_plot(data_name,df):
        # 选择数值型特征
        numeric_features = df.select_dtypes(include=['int', 'float'])
        # 计算偏态和峰态
        skewness = numeric_features.skew()
        kurtosis = numeric_features.kurtosis()
        # 创建画布
        plt.figure(figsize=(12, 5))
        # 绘制偏态图
        plt.subplot(1, 2, 1)
        sns.barplot(x=skewness.index, y=skewness.values, palette='viridis')
        plt.xticks(rotation=45)
        plt.title('数值型特征的偏度(%s数据集)'%data_name)
        plt.xlabel('特征')
        plt.ylabel('偏度')
        # 绘制峰态图
        plt.subplot(1, 2, 2)
        sns.barplot(x=kurtosis.index, y=kurtosis.values, palette='viridis')
        plt.xticks(rotation=45)
        plt.title('数值型特征的峰度(%s数据集)'%data_name)
        plt.xlabel('特征')
        plt.ylabel('峰度')
        plt.tight_layout()
        plt.show()

    data_name = 'titanic'
    nums_skew_kurt_plot(data_name,titanic_df)


if nums_distribution:
    # (2)、【数值型】特征分布性可视化
    def nums_distribution(data_name,df):
        # 选择数值型特征
        numeric_features = df.select_dtypes(include=['int', 'float'])
        # 可视化数值型特征的分布
        n_columns = 3
        n_rows = (len(numeric_features.columns) + n_columns - 1) // n_columns
        plt.figure(figsize=(12, 4 * n_rows))
        for i, feature in enumerate(numeric_features.columns):
            plt.subplot(n_rows, n_columns, i + 1)
            #  kde=True 表示同时绘制核密度估计曲线，bins=20 表示将数据分成 20 个箱子
            sns.histplot(data=numeric_features, x=feature, kde=True, bins=20, color='skyblue')
            plt.title(f'%s分布(%s数据集)' % (feature, data_name))
            plt.ylabel('频率')
            plt.xlabel(feature)
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    nums_distribution(data_name,titanic_df)


if nums_QQ_distribution:
    # (3)、【数值型】特征分布对比正态分布
    def nums_QQ_distribution(data_name,df):
        # 选择数值型特征
        numeric_features = df.select_dtypes(include=['int', 'float'])
        # 绘制 QQ分位数图
        n_columns = 3
        n_rows = (len(numeric_features.columns) + n_columns - 1) // n_columns
        plt.figure(figsize=(11, 4 * n_rows))
        import scipy.stats as stats
        for i, feature in enumerate(numeric_features.columns):
            plt.subplot(n_rows, n_columns, i + 1)
            stats.probplot(numeric_features[feature], dist="norm", plot=plt)
            plt.title(f'QQ图：%s分布(%s数据集)' % (feature, data_name))
        plt.tight_layout()
        plt.show()

    data_name = 'titanic'
    nums_QQ_distribution(data_name,titanic_df)


if num_transformed_distribution_PK:
    # (4)、【数值型】特征的长尾分布转为正态分布并对比
    def num_transformed_distribution_PK(data_name,df):
        # 选择数值型特征
        numeric_features = df.select_dtypes(include=['int', 'float']).copy()
        num_feature = 'fare'
        # 绘制Fare特征的直方图
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(numeric_features[num_feature], kde=True, color='skyblue')
        plt.title(f'直方图：%s分布(%s数据集)' % (num_feature, data_name))
        plt.xlabel(num_feature)
        # 对特征进行对数变换
        import numpy as np
        numeric_features['%s_log'%num_feature] = np.log1p(numeric_features[num_feature])
        

        # 绘制变换后Fare特征的直方图
        plt.subplot(1, 2, 2)
        sns.histplot(numeric_features['%s_log'%num_feature], kde=True, color='salmon')
        plt.title(f'直方图：%s执行log转换后的分布(%s数据集)' % (num_feature, data_name))
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    num_transformed_distribution_PK(data_name,titanic_df)


if num2_features:
    # 5.2.4.2、相关性(关联性分析)：
    # (1)、成对的【数值型】特征散点图关系可视化
    def num2_features(data_name,df,num_features):
        # 选择数值型特征
        numeric_features = df.select_dtypes(include=['int64', 'float64'])
        # 绘制散点图关系可视化
        sns.scatterplot(data=numeric_features, x=num_features[0], y=num_features[1])
        plt.title(f'散点图：%s和%s的关系分布(%s数据集)'%(num_features[0],num_features[1],data_name))
        plt.xlabel(num_features[0])
        plt.ylabel(num_features[1])
        plt.show()
    data_name = 'titanic'
    num_features = ['age','fare']
    num2_features(data_name,titanic_df,num_features)
    data_name = 'boston'
    num_features = ['LSTAT','B']
    num2_features(data_name,boston_df,num_features)


if num_features_heatmap:
    # (2)、整体所有的【数值型】特征热图关系可视化：PCC、MIC
    def num_features_heatmap(data_name,df):
        # 选择数值型特征
        numeric_features = df[['age', 'fare', 'sibsp', 'parch']].copy()
        numeric_features = numeric_features.fillna(numeric_features.median())
        # 计算PCC矩阵
        pcc_matrix = numeric_features.corr(method='pearson')
        # 绘制PCC皮尔森矩阵线性相关性热图
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(pcc_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title(f'热图：PCC可视化(%s数据集)' % (data_name))
        plt.xlabel('特征')
        plt.ylabel('特征')
        # 计算MIC矩阵
        from minepy import MINE
        import numpy as np
        mic_matrix = np.zeros((len(numeric_features.columns), len(numeric_features.columns)))
        for i, feature1 in enumerate(numeric_features.columns):
            for j, feature2 in enumerate(numeric_features.columns):
                mine = MINE()
                mine.compute_score(numeric_features[feature1], numeric_features[feature2])
                mic_matrix[i, j] = mine.mic()
        # 绘制MIC最大信息系数相关性热图
        plt.subplot(1, 2, 2)
        sns.heatmap(mic_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title(f'热图：MIC可视化(%s数据集)' % (data_name))
        plt.xlabel('特征')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    num_features_heatmap(data_name,titanic_df)


if nums_label_relation:
    # (3)、【数值型】特征与label之间的关系可视化
    def nums_label_relation(data_name,titanic_df,mix_features,label_name):
        # 单特征与label之间的关系可视化
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        sns.histplot(data=titanic_df, x=mix_features[0], hue=label_name, kde=True, palette='Set1')
        plt.title(f'%s和%s的关系分布(%s数据集)'%(mix_features[0],label_name,data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel('计数')
        # 组合特征与label之间的关系可视化
        plt.subplot(1, 3, 2)
        sns.scatterplot(data=titanic_df, x=mix_features[0], y=mix_features[1], hue=label_name, palette='Set2')
        plt.title(f'%s和%s的关系分布(%s数据集)'%(mix_features[0],mix_features[1],data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel(mix_features[1])
        # 统计特征与label之间的关系可视化
        plt.subplot(1, 3, 3)
        sns.barplot(data=titanic_df, x=mix_features[2], hue=label_name, y=mix_features[1], palette='Set3')
        plt.title(f'%s和%s的关系分布(%s数据集)'%(mix_features[2],mix_features[1],data_name))
        plt.xlabel(mix_features[2])
        plt.ylabel(mix_features[1])
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    mix_features = ['age','fare','pclass']
    nums_label_relation(data_name,titanic_df,mix_features,label_name)


# 5.2.4.2 数值型特征的关联分析：平行坐标图
if C_PCP_scaler_plot:
    # 分类任务的平行坐标图：只缩放输入列，不改变类别标签。
    def C_PCP_scaler_plot(data_name, df, label_name):
        from pandas.plotting import parallel_coordinates
        from sklearn.preprocessing import MinMaxScaler

        columns = ['age', 'fare', 'sibsp', 'parch']
        plot_data = df[columns + [label_name]].copy()
        plot_data[columns] = plot_data[columns].fillna(plot_data[columns].median())
        plot_data[label_name] = plot_data[label_name].astype(int).astype(str)
        plot_data = plot_data.sample(n=min(100, len(plot_data)), random_state=42)
        normalized = plot_data.copy()
        normalized[columns] = MinMaxScaler().fit_transform(plot_data[columns])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        parallel_coordinates(plot_data, label_name, cols=columns, ax=axes[0], alpha=0.35)
        parallel_coordinates(normalized, label_name, cols=columns, ax=axes[1], alpha=0.35)
        axes[0].set_title(f'原始特征平行坐标图（{data_name}）')
        axes[1].set_title(f'归一化后平行坐标图（{data_name}）')
        for ax in axes:
            ax.set_xlabel('特征')
            ax.set_ylabel('数值')
        fig.tight_layout()
        plt.show()

    C_PCP_scaler_plot('titanic', titanic_df, label_name)


if R_PCP_scaler_plot:
    # 回归任务的平行坐标图：连续标签用颜色表示，但不参与特征缩放。
    def R_PCP_scaler_plot(data_name, df, label_name):
        from matplotlib.colors import Normalize
        from sklearn.preprocessing import MinMaxScaler

        numeric = df.select_dtypes(include='number').drop(columns=label_name, errors='ignore')
        columns = numeric.columns[:6].tolist()
        sample = df[columns + [label_name]].copy()
        sample[columns] = sample[columns].fillna(sample[columns].median())
        sample[label_name] = pd.to_numeric(sample[label_name], errors='coerce')
        sample = sample.dropna(subset=[label_name])
        sample = sample.sample(n=min(100, len(sample)), random_state=42)
        scaled = sample[columns].copy()
        scaled[columns] = MinMaxScaler().fit_transform(sample[columns])
        colors = Normalize(vmin=sample[label_name].min(), vmax=sample[label_name].max())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, values, title in zip(
            axes, (sample[columns], scaled), ('原始特征', '归一化后特征')
        ):
            for (_, row), target in zip(values.iterrows(), sample[label_name]):
                ax.plot(columns, row.to_numpy(), color=plt.cm.RdYlBu(colors(target)), alpha=0.3)
            ax.set_title(f'{title}平行坐标图（{data_name}）')
            ax.set_xlabel('特征')
            ax.set_ylabel('数值')
            ax.tick_params(axis='x', rotation=35)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=colors, cmap='RdYlBu'),
            ax=axes, label='目标值', shrink=0.7,
        )
        plt.show()

    R_PCP_scaler_plot('boston', boston_df, 'target')


if distributio_histplot_mix_features:
    # 5.2.5、组合关联统计分析：针对【类别型】+【数值型】
    # 基于单变量(【数值型特征】)密度图的多层颜色(【类别型特征】)可视化
    def distributio_histplot_mix_features(data_name,df,mix_features):
        # 设置画布大小
        plt.figure(figsize=(11, 7))
        # 子图一：年龄与性别的分布
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x=mix_features[0], hue=mix_features[1], multiple='stack', kde=True, palette='husl')
        plt.title('%s和%s的关联统计分布(%s数据集)' % (mix_features[0], mix_features[1], data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel('计数')
        plt.subplot(2, 2, 2)
        sns.kdeplot(data=df, x=mix_features[0], hue=mix_features[1], multiple='stack', palette='husl', fill=True)
        plt.title('%s和%s的关联统计分布(%s数据集)'%(mix_features[0],mix_features[1],data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel('计数')
        # 子图二：年龄与生存情况的关系
        plt.subplot(2, 2, 3)
        sns.histplot(data=df, x=mix_features[0], hue=mix_features[2], multiple='stack', kde=True)
        plt.title('%s和%s的关联统计分布(%s数据集)'%(mix_features[0],mix_features[2],data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel('计数')
        plt.subplot(2, 2, 4)
        sns.kdeplot(data=df, x=mix_features[0], hue=mix_features[2], multiple='stack', fill=True)
        plt.title('%s和%s的关联统计分布(%s数据集)'%(mix_features[0],mix_features[2],data_name))
        plt.xlabel(mix_features[0])
        plt.ylabel('计数')
        plt.tight_layout()
        plt.show()
    data_name = 'titanic'
    mix_features = ['age','sex','target']
    distributio_histplot_mix_features(data_name,titanic_df,mix_features)


# 5.2.5.2 类别型与数值型特征的组合关联图
if distributio_jointplot_mix_features:
    # 带边缘分布的双变量(【数值型特征】)散点图多层颜色(【类别型特征】)可视化
    def distributio_jointplot_mix_features(data_name,df,mix_features):
        joint_plot = sns.jointplot(data=df, x=mix_features[0], y=mix_features[1], hue=mix_features[2],
                                   alpha=0.6, palette='husl')
        plt.show()
    data_name = 'titanic'
    mix_features = ['age','fare','target']
    distributio_jointplot_mix_features(data_name,titanic_df,mix_features)


if distributio_jointplot_all_features:
    # 整体矩阵图分布可视化
    def distributio_jointplot_all_features(data_name,df,label_name):
        # 绘制整体矩阵图
        columns = ['pclass', 'sex', 'age', 'fare', label_name]
        sns.pairplot(df[columns], hue=label_name, palette='husl')
        plt.show()
    data_name = 'titanic'
    distributio_jointplot_all_features(data_name,titanic_df,label_name)
