#"集成学习模型使用一系列弱学习器（也称为基础模型或基模型）进行学习，并将各个弱学习器的结果进行整合，从而获得比单个学习器更好的学习效果
#"集成学习模型的常见算法有Bagging算法和Boosting算法两种。
# Bagging算法的典型机器学习模型为随机森林模型，而Boosting算法的典型机器学习模型则为AdaBoost、GBDT、XGBoost和LightGBM模型

#1、Bagging算法
#Bagging算法的原理类似投票，每个弱学习器都有一票，最终根据所有弱学习器的投票，按照'少数服从多数'的原则产生最终的预测结果
#假设原始数据共有10000条，从中随机有放回地抽取10000次数据构成一个新的训练集（因为是随机抽样，所以可能出现某一条数据多次被抽中，
#也有可能某一条数据一次也没有被抽中），每次使用一个训练集训练一个弱学习器。这样有放回地随机抽取n次后，训练结束时就能获得由不同的
#训练集训练出的n个弱学习器，根据n个弱学习器的预测结果，按照'少数服从多数'的原则，获得一个更加准确、合理的预测结果
#具体来说，在分类问题中是用n个弱学习器投票的方式获取最终结果，在回归问题中则是取n个弱学习器的平均值作为最终结果


#2Boosting算法
#Boosting算法本质是将弱学习器提升为强学习器，
# 它和Bagging算发的区别在于：Bagging算法对待所有的弱学习器'一视同仁'；而Boosting算法'区别对待'通俗来说就是"培养精英"和"重视错误"
#'培养精英'就是每一轮训练后对预测结果较准确的弱学习器给予较大的权重，对表现不好的弱学习器则降低其权重。这样在最终端预测时，'优秀模型'的权重是大的，
#相当于它可以投出多票，而'一般模型'只能投出一票或不能投票。
#'重视错误'就是在每一轮训练后改变训练集的权值或概率分布，通过提高在前一轮被弱学习器对预测错误的样例的权值，
# 降低前一轮被学习器预测错误的数据的重视程度，从而提升模型整体预测效果