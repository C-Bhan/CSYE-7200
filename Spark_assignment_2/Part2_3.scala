import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Row
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.IndexToString



////////// Loading the csv files for the dataset /////////////////////
val train = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/train.csv")
val test = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/test.csv")
val preds = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/gender_submission.csv")

// FEATURE ENGINEERING

val dtrain1 = train.drop("Cabin")
val dtest1 = test.drop("Cabin")

val dtest2 = preds.join(test, Seq("PassengerId"), "full_outer")

val dtrain = dtrain1.withColumn("FamilySize", col("SibSp") + col("Parch"))
val dtest = dtest2.withColumn("FamilySize", col("SibSp") + col("Parch"))

val avgFareTrain = dtrain.agg(avg("Fare")).head.getDouble(0)
val avgAgeTrain = dtrain.agg(avg("Age")).head.getDouble(0)
val avgFareTest = dtest.agg(avg("Fare")).head.getDouble(0)
val avgAgeTest = dtest.agg(avg("Age")).head.getDouble(0)
val embarkedTrain = udf((s: String) => Option(s).filter(_.nonEmpty).getOrElse("S"))
val embarkedTest = udf((s: String) => Option(s).filter(_.nonEmpty).getOrElse("S"))


val dprune1 = dtrain.na.fill(Map("Age" -> avgAgeTrain, "Fare" -> avgFareTrain))
val dprunetrain = dprune1.withColumn("Embarked", embarkedTrain(dprune1.col("Embarked")))

val dprune2 = dtest.na.fill(Map("Age" -> avgAgeTest, "Fare" -> avgFareTest))
val dprunetest = dprune2.withColumn("Embarked", embarkedTrain(dprune2.col("Embarked")))


val catcols = Seq("Pclass", "Sex", "Embarked")
val featcols = Seq("Age", "SibSp", "Parch", "Fare", "FamilySize")

val allfeatcols = featcols ++ catcols.map(_ + "Index")
val stringIndexer = catcols.map { colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "Index").fit(dprunetrain)}
val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("SurvivedIndex").fit(dprunetrain)

val vectorassembler = new VectorAssembler().setInputCols(Array(allfeatcols: _*)).setOutputCol("Features")


// MODEL


val randomForestClassifier = new RandomForestClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("Features")

val createlabels = new IndexToString().setInputCol("prediction").setOutputCol("predicted").setLabels(labelIndexer.labels)

// MODEL PARAMETERS

val paramMap = new ParamGridBuilder().addGrid(randomForestClassifier.impurity, Array("gini", "entropy")).addGrid(randomForestClassifier.maxDepth, Array(1,2,5, 10, 15)).build()

val eval = new BinaryClassificationEvaluator().setLabelCol("SurvivedIndex")

// Pipeline

val pipe = new Pipeline().setStages((stringIndexer :+ labelIndexer :+ vectorassembler :+ randomForestClassifier :+ createlabels).toArray)


// Validator

val crossval = new CrossValidator().setEstimator(pipe).setEvaluator(eval).setEstimatorParamMaps(paramMap).setNumFolds(10)

// Training

val crossvalmodel = crossval.fit(dprunetrain)

val preds = crossvalmodel.transform(dprunetest)

print("ACCURACY: ")
preds.select(avg(($"Survived" === $"predicted").cast("integer"))).show()


















