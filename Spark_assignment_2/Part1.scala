
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Row
import breeze.linalg.{DenseMatrix, DenseVector}

////////// Loading the csv files for the dataset /////////////////////
val dtrain = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/train.csv")
val dtest = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/test.csv")
val dpreds = spark.read.option("header", "true").option("inferSchema", "true").csv("./titanic/gender_submission.csv")


////// Combine the rows adding survived to test data ///////

val dtestp = dpreds.join(dtest, Seq("PassengerId"), "full_outer")

///////Combining train and test dataset//////////
val data = dtrain.union(dtestp)

//////// Data schema ///////
data.printSchema()
///////// Dataset ///////////
data.show()

val droppeddata = data.drop("Name","Sex","Ticket","Embarked", "Cabin")

// val corrmatrix = numdata.stat.corr("age", "fare", "pclass", "sibsp", "parch", method = "pearson")



// Handle NULL values:

val avgFare = data.agg(avg("Fare")).head.getDouble(0)
val avgAge = data.agg(avg("Age")).head.getDouble(0)


val numdata = droppeddata.na.fill(Map("Age" -> avgAge, "Fare" -> avgFare))
val fulldata = data.na.fill(Map("Age" -> avgAge, "Fare" -> avgFare))

// FINDING THE CORRELATION BETWEEN DATA:

val selectedCols = Array("Survived", "Pclass", "Age", "SibSp", "Parch", "Fare")

val assembler = new VectorAssembler()
  .setInputCols(selectedCols)
  .setOutputCol("features")

val featureDF = assembler.transform(numdata).select("features")

// Calculate the Pearson correlation matrix
val pearsonCorr = Correlation.corr(featureDF, "features", "pearson").head match {
  case Row(matrix: Matrix) => matrix
}

println("Pearson correlation matrix:")
println(selectedCols.mkString("\t\t\t"))
println(pearsonCorr.toString(10,500))

// Calculate the Spearman correlation matrix
val spearmanCorr = Correlation.corr(featureDF, "features", "spearman").head match {
  case Row(matrix: Matrix) => matrix
}

println("Spearman correlation matrix:")
println(selectedCols.mkString("\t\t\t"))
println(spearmanCorr.toString(10,500))

numdata.describe().show()

// SOME MORE STATISTICS:

numdata.groupBy("Survived").count().show()

fulldata.groupBy("Sex", "Survived").agg(count("*").alias("count")).orderBy("Sex", "Survived").show()

fulldata.groupBy("Pclass", "Survived").agg(count("*").alias("count")).orderBy("Pclass", "Survived").show()

//////// Method to find the survivors % in each ticket class /////////////
def survivorsByPclass(pclass:Long, dataset:DataFrame): Double = {

    val result  = ((dataset.filter(dataset("Pclass") === pclass.toString && dataset("Survived") === "1").count()).toDouble / (dataset.filter(dataset("Pclass") === pclass.toString).count()).toDouble)*100

    result.toDouble
}

def survivorsBySex(sex:String, dataset:DataFrame): Double = {

    val result  = ((dataset.filter(dataset("Sex") === sex.toString && dataset("Survived") === "1").count()).toDouble / (dataset.filter(dataset("Sex") === sex.toString).count()).toDouble)*100

    result.toDouble
}


def avgFareByClass(pclass:Long, dataset:DataFrame): Double = {

    val result = ((dataset.filter(dataset("Pclass") === pclass.toString).agg(sum("Fare")).head().getDouble(0)) / (dataset.filter(dataset("Pclass") === pclass).count()).toDouble)

    result.toDouble

}


/////// Average Fare price by ticket class //////////////

println(s"1st Class average ticket prices: ${avgFareByClass(1, fulldata)}")
println(s"2nd Class average ticket prices: ${avgFareByClass(2, fulldata)}")
println(s"3rd Class average ticket prices: ${avgFareByClass(3, fulldata)}")


/////////// Surviviors by ticket class //////////////////
println(s"Percentage of 1st Class survivors: ${survivorsByPclass(1, fulldata)}%")
println(s"Percentage of 2nd Class survivors: ${survivorsByPclass(2, fulldata)}%")
println(s"Percentage of 3rd Class survivors: ${survivorsByPclass(3, fulldata)}%")

println(s"Percentage of Male survivors: ${survivorsBySex("male", fulldata)}%")
println(s"Percentage of Female survivors: ${survivorsBySex("female", fulldata)}%")

////////// Dataset for Ageclass column along with Fare and Survived /////////////////////////
val ageClassData = fulldata.select(col("Age"), col("Fare"), col("Survived")).withColumn("AgeClass",
    when(col("Age").between(1, 10), "1to10")
    .when(col("Age").between(11, 20), "11to20")
    .when(col("Age").between(21, 30), "21to30")
    .when(col("Age").between(31, 40), "31to40")
    .when(col("Age").between(41, 50), "41to50")
    .when(col("Age").between(51, 60), "51to60")
    .when(col("Age").between(61, 70), "61to70")
    .when(col("Age").between(71, 80), "71to80")
    .otherwise(null)
)

ageClassData.show()


//////// Method to get
def ageClassStats(ageclass:String): DataFrame = {

    val result = ageClassData.filter(ageClassData("AgeClass") === ageclass).describe("Fare","Survived")

    result
}


//////////// Getting Fare and survivor stats for different Age classes //////////////////////
println("Details for Age Class [1 - 10]:" )
println(ageClassStats("1to10").show())
println("Details for Age Class [11 - 20]: ")
println(ageClassStats("11to20").show())
println("Details for Age Class [21 - 30]: ")
println(ageClassStats("21to30").show())
println("Details for Age Class [31 - 40]: ")
println(ageClassStats("31to40").show())
println("Details for Age Class [41 - 50]: ")
println(ageClassStats("41to50").show())
println("Details for Age Class [51 - 60]: ")
println(ageClassStats("51to60").show())
println("Details for Age Class [61 - 70]: ")
println(ageClassStats("61to70").show())
println("Details for Age Class [71 - 80]: ")
println(ageClassStats("71to80").show())

