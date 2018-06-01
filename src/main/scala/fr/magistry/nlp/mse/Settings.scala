package fr.magistry.nlp.mse

import java.io.File

import com.typesafe.config.ConfigException.{Missing, WrongType}
import com.typesafe.config.ConfigFactory
import fr.magistry.nlp.datatypes._
import fr.magistry.nlp.mse
import fr.magistry.nlp.utils.Mimick

/**
  * Singleton object to handle configuration options
  */
object Settings {
  private val cfg = ConfigFactory.load()

  private def wrapErrorReport[T](f: String => T, key: String, msg: String): T = {
    try f(key)
    catch {
      case e: Missing => println("missing option: " + msg) ; throw e
      case e: WrongType => println("wrongtype for option: " + msg) ; throw e
    }
  }


  val python2: String = wrapErrorReport(cfg.getString, "env.python2", "python2 command (for mimick)")
  val python3: String = wrapErrorReport(cfg.getString, "env.python3", "python3 command(for morfessor, MSE and yaset)")

  object tokenizer {
    val strategy: Options.CorpusLoadingFashion = wrapErrorReport(
      { key =>
        cfg.getString(key) match {
          case "conll" => Options.ReadConll
          case "external" => Options.ExternalTokenizer
  //        case "opennlp" => Options.CallOpenNLPTokenizer
          case _ => throw new IllegalArgumentException
        }
      },
      "tokenizer.strategy",
      "tokenizer strategy"
    )

   // lazy val path: String = wrapErrorReport(cfg.getString, "tokenizer.path", "no file to read")
    lazy val command: String = wrapErrorReport(cfg.getString, "tokenizer.command", "no external command for tokenization")
    lazy val model: String = wrapErrorReport(cfg.getString, "tokenizer.model", "no opennlp model for tokenization")
  }

  object corpora {
    lazy val trainingSet: GoldCorpus =
      wrapErrorReport(
        {key => Corpora.loadGoldCorpus(cfg.getString(key))},
        "corpora.training",
        "training corpus path"
      )

    lazy val testingSet: GoldCorpus =
      wrapErrorReport(
        {key => Corpora.loadGoldCorpus(cfg.getString(key))},
        "corpora.testing",
        "testing corpus path"
      )

    lazy val raw: CorpusAnnotated =
      wrapErrorReport(
        {key =>
          val path = cfg.getString(key)
          tokenizer.strategy match {
          case Options.ReadConll => Corpora.loadConllCorpus(path)
          case Options.ExternalTokenizer =>
              val t = new ExternalTokenizer(tokenizer.command)
              Corpora.dummyTagger(t(path))
          }
        },
        "corpora.raw",
        "no raw corpus specified"
      )
  }


  object embeddings {

    lazy val file: File =
      wrapErrorReport(
        { key => new File(cfg.getString(key)) },
        "embeddings.file",
        "pb with embeddings file configuration"
      )

    lazy val ndim: Int =
      wrapErrorReport(
        { key => cfg.getInt(key) },
        "embeddings.ndim",
        "number of dimensions for embeddings"
      )
    lazy val ws: Int =
      wrapErrorReport(
        { key => cfg.getInt(key) },
        "embeddings.ws",
        "window size for embeddings"
      )
    lazy val nbOccMin: Int =
      wrapErrorReport(
        { key => cfg.getInt(key) },
        "embeddings.nboccmin",
        "minimum number of occurrences for embeddings"
      )

    lazy val oovFile: File =
      wrapErrorReport(
        { key => new File(cfg.getString(key)) },
        "embeddings.oovfile",
        "problem with oov file configuration"
      )
  }

  object yaset {
    lazy val model: File =
      wrapErrorReport(
        { key => new File(cfg.getString(key)) },
        "yaset.model",
        "model zip file not specified"
      )
  }


}