package fr.magistry.nlp.mse

import java.io.{File, FileWriter, OutputStreamWriter}

import fr.magistry.nlp.Eval
import fr.magistry.nlp.datatypes.CorpusAnnotated
import fr.magistry.nlp.utils.{Mimick, Morfessor, Yaset}

import scala.io.Source

object Main extends App {


  def parseArgs(options: Map[String, String], args: List[String], toRead: List[String]): (Map[String,String], List[String]) = {
    toRead match {
      case Nil =>
        (options, toRead)
      case "--loop" :: i :: tail =>
        parseArgs(options.updated("loop", i), args, tail)
      case "--learn-embeddings" :: tail =>
        parseArgs(options.updated("action", "learn-embeddings"), args, tail)
      case "--train-yaset" :: tail =>
        parseArgs(options.updated("action", "train-yaset"), args, tail)
      case "--apply-yaset" :: tail =>
        parseArgs(options.updated("action", "apply-yaset"), args, tail)
      case "--split" :: src :: prop :: a :: b :: tail =>
        parseArgs(
          options
            .updated("action", "split")
            .updated("src", src)
            .updated("prop", prop)
            .updated("a", a)
            .updated("b", b),
          args,
          tail
        )
      case "--eval" :: tail =>
        parseArgs(options.updated("action", "eval"), args, tail)
      case _ =>
        println("unparsed argument"); sys.exit() //throw new IllegalArgumentException
    }
  }


  def getEmbeddings(): Embeddings = {
    if(Settings.embeddings.file.exists()) {
      MorphoSyntacticEmbeddings.loadFromSettings()
    }
    else {
      trainEmbeddings()
    }
  }


  lazy val vocabulary = {
    val corpora = Seq(
      Settings.corpora.raw,
      Settings.corpora.trainingSet,
      Settings.corpora.testingSet
    ) ++ Settings.corpora.devSet.map(Seq(_)).getOrElse(Seq.empty)
    corpora
      .map(Corpora.vocabularyOfCorpus(_))
      .foldLeft(Set.empty[String]){_ union _}
  }

  def trainEmbeddings(raw: CorpusAnnotated=Settings.corpora.raw,
                      morpho: Option[Map[String, Set[String]]]=None): Embeddings = {
    val mimick = new Mimick(Settings.python2)
    val mse = MorphoSyntacticEmbeddings.initFromSettings(mimick)
    val oovSrc = scala.io.Source.fromFile(Settings.embeddings.oovFile)
    val targetVoc = oovSrc.getLines().map(_.trim).toSet union vocabulary
    oovSrc.close()
    mse.train(raw, morpho)
    mse.completeVocabulary(targetVoc)
    mse
  }

  def trainYaset(mse: Embeddings): Yaset = {
    val yaset = new Yaset("/tmp/")
    yaset.train(mse, Settings.corpora.trainingSet, Settings.corpora.devSet)
    yaset
  }

  def trainEmbeddingsLoop(n: Int): Embeddings = {
    val morpho = new Morfessor().buildMorphoMapping(Settings.corpora.raw, None, false)
    val mse0 = trainEmbeddings(morpho=Some(morpho))
    (1 to n).foldLeft(mse0) { (mse,_) =>
      val yaset = trainYaset(mse)
      val raw2 = yaset(Settings.corpora.raw)
      Settings.corpora.retaggedRawPath.foreach {Corpora.writeConll(raw2,_)}
      trainEmbeddings(raw2, Some(morpho))
    }
  }

  override def main(args: Array[String]): Unit = {
    val (options, arguments) = parseArgs(Map.empty, Nil, args.toList)
    options.get("action") match {
      case None => println("no action defined")
      case Some("learn-embeddings") =>
        val nloop = options.get("loop").map(_.toInt).getOrElse(0)
        val outputFile = Settings.embeddings.file
        println(s"learn embeddings from rawdata in ${Settings.corpora.raw.name} read by  ${Settings.tokenizer.strategy}, in $nloop loop(s)")
        val mse = trainEmbeddingsLoop(nloop)
        mse.toFile(outputFile)
      case Some("train-yaset") =>
        val outputFile = Settings.yaset.model
        val mse = getEmbeddings() // read or train
        val yaset = trainYaset(mse)
        yaset.saveModel(outputFile.getAbsolutePath)
      case Some("apply-yaset") =>
        val testCorpus = Settings.corpora.testingSet
        if(! Settings.yaset.model.exists()) {
          println("train a model first")
          sys.exit()
        }
        val modelPath = Settings.yaset.model.getAbsolutePath
        val retaggedCorpus = Yaset.applyPretrained(modelPath, testCorpus)
        Corpora.writeConll(retaggedCorpus, Settings.corpora.output)
      case Some("eval") =>
        val goldCorpus = Settings.corpora.testingSet
        val testCorpus = Yaset.applyPretrained(Settings.yaset.model.getAbsolutePath, goldCorpus)
        val e = new Eval(goldCorpus, testCorpus)
        println(e.confusion)
        println(e.accuracy)
      case Some("split") =>
        val src = Corpora.loadConllCorpus(options("src"))
        val (a,b) = Corpora.splitAnnotatedCorpus(src, options("prop").toDouble)
        Corpora.writeConll(a, options("a"))
        Corpora.writeConll(b, options("b"))
      case Some(a) => println(s"unrecognised action $a")
    }
  }

}
