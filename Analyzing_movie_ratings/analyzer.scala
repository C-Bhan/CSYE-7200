
import org.apache.spark.sql.functions.{avg, stddev}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import scala.util.Try


// Analyzer Code

val movie_data = input("/home/bhan/College/Scala/code/CSYE7200/spark-csv/src/main/resources/movie_metadata.csv", spark)

movie_data.show()

val mean_data = analyze(movie_data, "imdb_score", "mean")
val stddev_data = analyze(movie_data, "imdb_score", "std_deviation")

mean_data.show()
stddev_data.show()




def input(path: String, session: SparkSession): DataFrame = {

    session.read.option("header", "true").option("inferSchema", "true").csv(path)

}

def analyze(data: DataFrame, rating: String, stat: String): DataFrame = {

    stat match{

    case "mean" => data.select(avg(rating))
    case "std_deviation" => data.select(stddev(rating))

    }

}

// Tests:

val list = List(2.5, 7.7, 4.1, 3.9)

val meandf = analyze(list.toDF(), "value", "mean" )
val stddevdf = analyze(list.toDF(), "value", "std_deviation" )

if(meandf.first().getDouble(0) == 4.55){print("Mean test OK \n")}
if(stddevdf.first().getDouble(0) == 2.217355782608345){print("Standard deviation test OK \n")}



