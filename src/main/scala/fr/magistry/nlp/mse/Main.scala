package fr.magistry.nlp.mse

import java.io.{File, FileWriter, OutputStreamWriter}

import fr.magistry.nlp.utils.{Mimick, Yaset}

import scala.io.Source

object Main extends App {


  def parseArgs(options: Map[String, String], args: List[String], toRead: List[String]): (Map[String,String], List[String]) = {
    toRead match {
      case Nil => (options, toRead)
      case "--learn-embeddings" :: file :: tail => parseArgs(options.updated("action", "learn-embeddings").updated("output", file), args, tail)
      case "--train-yaset" :: file :: tail => parseArgs(options.updated("action", "train-yaset").updated("output", file), args, tail)
      case _ => println("unparsed argument"); throw new IllegalArgumentException
    }
  }


  def getEmbeddings(): Embeddings = {
    val mimick = new Mimick(Settings.python2)
    val mse = new MorphoSyntacticEmbeddings(Settings.python3, 50, 1, 1, true, mimick)
    mse.train(Settings.corpora.raw)
    mse
  }

  override def main(args: Array[String]): Unit = {
    val (options, arguments) = parseArgs(Map.empty, Nil, args.toList)
    options.get("action") match {
      case None => println("no action defined")
      case Some("learn-embeddings") =>
        val outputFile = new File(options("output"))
        assert(!outputFile.exists())
        println(s"learn embeddings from rawdata in ${Settings.corpora.raw.name} read by  ${Settings.tokenizer.strategy}")
        val mse = getEmbeddings()
        mse.toFile(outputFile)
      case Some("train-yaset") =>
        val outputFile = new File(options("output"))
        assert(!outputFile.exists())
        val mse = getEmbeddings()
        val yaset = new Yaset("/tmp/")
        yaset.train(mse, Settings.corpora.trainingSet, None)
      case Some(a) => println(s"unrecognised action $a")
    }
  }

}
