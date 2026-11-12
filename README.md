# 股票价格区间预测实验代码

## 主实验
入口为 run_ablation_no_cross_attention.py。文件名沿用开发阶段命名。
选择依据是 V4 的价格与新闻双 GRU、市场状态标量门控、残差融合和中心／半径输出结构，以及后续绘图、稳健性和解释性脚本对 outputs_ablation_no_cross_attention 的依赖。
原包内 stockv2_final_code.zip 对应带交叉注意力的旧版本。

## 实验入口
- 对比：run_ptca_final.py。原包未定位到独立 GRU、PatchTST、FIVMD-LSTM 基线脚本。
- 消融：run_ablation_price_only.py、run_ablation_no_market_state.py、run_ablation_no_cross_no_gate.py、run_ablation_news_representation.py。
- 稳健性：窗口长度、市场状态、新闻条件 fixed 版、新闻滞后期。
- 绘图与可解释性：plot_fig2_fig3_final_model.py 和四个 run_interpretability_* 脚本。绘图文件名中的图号沿用开发阶段命名。
- 公共依赖：postexp_utils.py、run_stockv2_tune.py、stockv2_data_tune.py、stockv2_model_tune.py。

## 运行
安装 numpy、pandas、torch、scikit-learn、matplotlib 后，从代码目录执行：
    python run_ablation_no_cross_attention.py --help
    python run_ablation_no_cross_attention.py --root . --horizons 1,3,5 --device cpu

输入包括 sh_index.csv、sz_index.csv、结构化新闻 CSV 和相匹配的 1024 维嵌入 NPZ。使用原实验的配对文件，并按脚本中的输入文件识别规则命名。部分后续脚本默认使用 Windows 本机路径，运行时传入实际 --root。解释性脚本需要训练产生的检查点或预测结果。

## 复现核对
此包按源码结构及调用关系筛选。原压缩包未包含运行日志、检查点和逐项实验输出，尚未逐项重跑确认与论文表格数值相同。尤其应核对表8：fixed 脚本对三类新闻属性加噪，而 V4 文字描述为只扰动情绪得分。
