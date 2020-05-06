# quant_feature_engineering
 feature engineering for quant

用于生成资产的特征工程项目，主要输入数据为价量数据，输出为短期涨跌或者是短期相对大盘的涨跌(超额收益)

输入数据格式为dataframe,索引为date,字段为open	high	low	close	volume

调用示例：
import feature_engineering as fea_eng

#生成特征函数列表

feature_funcs=fea_eng.my_features_functions()

my_func_name_list=feature_funcs.get_all_methold()

func_list=[]
for func_name in my_func_name_list:
    func_list.append(eval("feature_funcs."+func_name))

#生成特征,25是滚动天数

my_fea_eng=fea_eng.feature_engineering(func_list,25)
pip_df_result=my_fea_eng.output_feature(data)
