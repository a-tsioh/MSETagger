package fr.magistry.nlp.utils

import java.io.{File, FileWriter}

import scala.io.Source
import scala.sys.process
import java.nio.file.Files

import fr.magistry.nlp.mse.Embeddings
import fr.magistry.nlp.datatypes._

class Yaset(workdir: String,
            metric: String="accuracy",
            updateEmbeddings: Boolean=true,
            charEmbeddings: Boolean=true,
            hlSize: Int=12,
            batchSize: Int=16,
            charEmbeddingsSize: Int = 8,
            charLayerSize: Int = 9,
            devRatio: Double = 0.2,
            name: String="name",
            useLast: Boolean = false
           ) {

  val cmd = "yaset"

  var model: Option[File] = None

  val root = new File(workdir)
  assert(root.isDirectory)

  private var _embeddings: Option[Embeddings] = None

  def getEmbeddings(): Option[Embeddings] = _embeddings

  private def dump2colCorpus(c:Corpus, path: String): Unit = {
    val inputFW = new FileWriter(new File(path))
    (c match {
      case g: CorpusAnnotated =>
        g.content.par.map { sent =>
          sent.tokens.map { case t: TokenWithPOS => s"${t.form}\t${t.pos}" }
        }
      case r: CorpusRaw =>
        r.content.par.map { sent =>
          sent.tokens.map { t =>
            s"${t.form}\t_"
          }
        }
    }).map { s=> ( s mkString "\n") + "\n\n" }
      .seq
      .foreach(inputFW.write)
    inputFW.close()
  }

  def train(embeddings: Embeddings, train: GoldCorpus, devOpt: Option[GoldCorpus]): Unit ={
    _embeddings = Some(embeddings)
    val embFile = new File(workdir,"yaset-emb")
    embeddings.toFile(embFile)
    this.train(embFile.getAbsolutePath, train, devOpt)
  }

  def train(embeddings: String, train: GoldCorpus, devOpt: Option[GoldCorpus]=None): Unit ={
    val workDir = File.createTempFile("restaureTmp","Yaset")
    workDir.delete()
    workDir.mkdir()

    val trainFile = new File(workDir, "train.pos")
    dump2colCorpus(train, trainFile.getAbsolutePath)

    val devFileOpt = devOpt.map { dev =>
      val devFile = new File(workDir, "dev.pos")
      dump2colCorpus(dev, devFile.getAbsolutePath)
      devFile
    }

    val configFile = new File(workDir, "config.ini")
    val configFW = new FileWriter(configFile)
    configFW.write(config(embeddings, trainFile.getAbsolutePath, devFileOpt.map(_.getAbsolutePath)))
    configFW.close()

    val args = Seq( cmd, "LEARN", "--config", configFile.getAbsolutePath)
    process.Process.apply(args).!

    model = Some(
      root
        .listFiles()
        .filter {f => f.isDirectory }
        .maxBy(_.lastModified)
    )
  }

  def trainFromFiles(embeddings: String, train: String, devOpt: Option[String]=None): Unit = {
    val workDir = File.createTempFile("restaureTmp","Yaset")
    workDir.delete()
    workDir.mkdir()

    val configFile = new File(workDir, "config.ini")
    val configFW = new FileWriter(configFile)
    configFW.write(config(embeddings, train, devOpt))
    configFW.close()
    val args = Seq( cmd, "LEARN", "--config", configFile.getAbsolutePath)
    process.Process.apply(args).!

    model = Some(
      root
        .listFiles()
        .filter {f => f.isDirectory }
        .maxBy(_.lastModified)
    )
  }

  //  def apply(inputData: RDD[Sentence]): RDD[Sentence] = {
  //    val inCorpus = new GoldCorpus(inputData.sparkContext,"") {
  //      override def goldContent(): RDD[Sentence] = inputData
  //    }
  //    val outCorpus = apply(inCorpus)
  //    outCorpus.goldContent()
  //  }

  def apply(corpus: Corpus): CorpusAnnotated = {
    val workDir = File.createTempFile("restaureTmp", "Yaset")
    workDir.delete()
    workDir.mkdir()

    val testFile = new File(workDir, "test.pos")
    dump2colCorpus(corpus, testFile.getAbsolutePath)

    val args = Seq(cmd, "APPLY", "--input-file", testFile.getAbsolutePath, "--working-dir", workdir, "--model-path", model.get.getAbsolutePath)
    process.Process.apply(args).!

    val resultDir =
      root
        .listFiles()
        .filter(f => f.isDirectory && f.getName.contains("apply"))
        .maxBy(_.lastModified)

    val outputFile = new File(resultDir, "output-model-001.conll")
    readCorpus(outputFile)
    //restaure.utils.rmdir(workDir)
  }

  def apply(test: String, output: String): Unit = {

    val args = Seq(cmd, "APPLY", "--input-file", test, "--working-dir", workdir, "--model-path", model.get.getAbsolutePath)
    process.Process.apply(args).!

    val resultDir =
      root
        .listFiles()
        .filter(f => f.isDirectory && f.getName.contains("apply"))
        .maxBy(_.lastModified)

    val outputFile = new File(resultDir, "output-model-001.conll")
    Files.copy(outputFile.toPath, new File(output).toPath)
  }


  def readCorpus(file: File): CorpusAnnotated = {
    val src = Source.fromFile(file)
    val stream = src.getLines().toStream
    def readStream(stream: Stream[String], acc:List[TokenWithPOS]): Stream[SentenceAnnotated] = {
      stream match {
        case Stream.Empty =>
          if(acc.isEmpty) Stream.Empty
          else SentenceAnnotated(acc.reverse.toArray) #:: Stream.Empty
        case line #:: tail if line.trim.isEmpty => SentenceAnnotated(acc.reverse.toArray) #:: readStream(tail, Nil)
        case line #:: tail =>
          val fields = line.trim.split("\t")
          val form = fields.head
          val tag = fields.last
          readStream(tail, TokenWithPOS(form, tag) :: acc)
      }
    }

    val data = readStream(stream, Nil).toArray
    src.close()

    CorpusAnnotated(
      "yaset-result",
      data
    )
  }





  def config(embeddings: String, train: String, dev: Option[String]): String =
    s"""
       |[general]
       |
       |batch_mode = false
       |
       |batch_iter = 5
       |
       |experiment_name = $name
       |
       |[data]
       |
       |train_file_path = $train
       |dev_file_use = ${dev.nonEmpty}
       |dev_file_path = ${dev.getOrElse("unused")}
       |dev_random_ratio = $devRatio
       |dev_random_seed_use = false
       |dev_random_seed_value = 42
       |
       |preproc_lower_input = false
       |preproc_replace_digits = false
       |feature_data = false
       |feature_columns = 1,2,3
       |
       |embedding_model_type = word2vec
       |embedding_model_path = $embeddings
       |embedding_oov_strategy = map
       |embedding_oov_map_token_id = ${"$$UNK$$"}
       |embedding_oov_replace_rate = 0.0
       |working_dir = $workdir
       |
       |[training]
       |
       |model_type = bilstm-char-crf
       |max_iterations = 200
       |patience = 75
       |store_matrices_on_gpu = true
       |bucket_use = false
       |
       |dev_metric = $metric
       |
       |trainable_word_embeddings = $updateEmbeddings
       |# NUmber of CPU cores to use during training
       |cpu_cores = 20
       |batch_size = $batchSize
       |
       |use_last = $useLast
       |
       |opt_algo = adam
       |opt_lr = 0.001
       |opt_gc_use = false
       |opt_gc_type = clip_by_norm
       |opt_gs_val = 5.0
       |
       |opt_decay_use = false
       |opt_decay_rate = 0.9
       |opt_decay_iteration = 1
       |
       |feature_use = false
       |feature_embeddings_size = 10
       |
       |[bilstm-char-crf]
       |
       |hidden_layer_size = $hlSize
       |dropout_rate = 0.5
       |
       |use_char_embeddings = $charEmbeddings
       |char_hidden_layer_size = $charLayerSize
       |char_embedding_size = $charEmbeddingsSize
    """.stripMargin





}
