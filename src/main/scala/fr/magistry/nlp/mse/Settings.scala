package fr.magistry.nlp.mse

import java.io.File

import com.typesafe.config.ConfigException.{Missing, WrongType}
import com.typesafe.config.ConfigFactory
import fr.magistry.nlp.datatypes._
import fr.magistry.nlp.mse
import fr.magistry.nlp.tokenization.BasicTokenizer
import fr.magistry.nlp.utils.Mimick

import scala.util.{Failure, Success}

/**
  * Singleton object to handle configuration options
  */
object Settings {
  private val cfg = ConfigFactory.load()

  private def wrapErrorReport[T](f: String => T, key: String, msg: String): T =
    scala.util.Try(f(key)) match {
      case Success(v) => v
      case Failure(e: Missing) => println("missing option: " + msg) ; sys.exit()
      case Failure(e: WrongType) => println("wrongtype for option: " + msg) ; sys.exit()
      }


  private def fileExistsOrExit(f:File): File = {
    if(!f.exists) {
      println(s"${f.getAbsolutePath} not found")
      sys.exit()
    }
    f
  }


  val python2: String = wrapErrorReport(cfg.getString, "env.python2", "python2 command (for mimick)")
  val python3: String = wrapErrorReport(cfg.getString, "env.python3", "python3 command(for morfessor, MSE and yaset)")

  object tokenizer {
    val strategy: Options.CorpusLoadingFashion = wrapErrorReport(
      { key =>
        cfg.getString(key) match {
          case "conll" => Options.ReadConll
          case "external" => Options.ExternalTokenizer
          case "basic" => Options.BasicTokenizer
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

    lazy val devSet: Option[GoldCorpus] =
      util.Try({
        val path = cfg.getString("corpora.dev")
        Corpora.loadGoldCorpus(path)
      }).toOption






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
          case Options.BasicTokenizer =>
              Corpora.dummyTagger(BasicTokenizer.fromFile(path))
          }
        },
        "corpora.raw",
        "no raw corpus specified"
      )

    lazy val retaggedRaw: Option[CorpusAnnotated] =
      util.Try(cfg.getString("corpora.retagged_raw")) match {
        case Success(path) => Some(Corpora.loadConllCorpus(path))
        case Failure(_) => None
      }

    lazy val retaggedRawPath: Option[String] =
      util.Try(cfg.getString("corpora.retagged_raw")).toOption


    lazy val output: String =
      wrapErrorReport(
        cfg.getString,
        "corpora.output",
        "no output corpus file specified"
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
        cfg.getInt,
        "embeddings.ndim",
        "number of dimensions for embeddings"
      )
    lazy val ws: Int =
      wrapErrorReport(
        cfg.getInt,
        "embeddings.ws",
        "window size for embeddings"
      )
    lazy val nbOccMin: Int =
      wrapErrorReport(
        cfg.getInt,
        "embeddings.nboccmin",
        "minimum number of occurrences for embeddings"
      )

    lazy val oovFile: File =
      fileExistsOrExit(
        wrapErrorReport(
        { key => new File(cfg.getString(key)) },
        "embeddings.oovfile",
        "problem with oov file configuration"
        )
      )
  }

  object yaset {
    val model: File =
      wrapErrorReport(
        { key => new File(cfg.getString(key)) },
        "yaset.model",
        "model zip file not specified"
      )
    object config {
      val useCharEmbeddings: Boolean =
        wrapErrorReport(
          cfg.getBoolean,
          "yaset.config.use_char_embeddings",
          "usage of character embeddings not specified"
        )

      val hlSize: Int =
        wrapErrorReport(
          cfg.getInt,
          "yaset.config.hl_size",
          "hl_size not specified or invalid"
        )

      val batchSize: Int =
        wrapErrorReport(
          cfg.getInt,
          "yaset.config.batch_size",
          "batch_size not specified or invalid"
        )

      val devRatio: Double =
        wrapErrorReport(
          cfg.getDouble,
          "yaset.config.dev_ratio",
          "dev_ratio not specified or invalid"
        )
    }
  }


}
