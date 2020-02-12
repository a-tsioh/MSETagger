package fr.magistry.nlp.mse


import java.io.{File, FileWriter}

import fr.magistry.nlp.datatypes._
import fr.magistry.nlp.utils.{Mimick, Morfessor}

import scala.io.Source
import scala.sys.process

class MorphoSyntacticEmbeddings(python3: String,
                                nbDim: Int,
                                nOccMin: Int,
                                ws: Int,
                                lowerCase: Boolean,
                                mimick: Mimick) extends Embeddings(nbDim, nOccMin, ws, lowerCase) {

  private val MORPH = "\ue00b"

  private def writePythonScriptToTemp(dir: File): String = {
    assert(dir.isDirectory)
    (for(name <- Seq("features.py","loaders.py", "train_keras.py")) yield {
      val stream = this.getClass.getResourceAsStream("/python/mse/" + name)
      val src = Source.fromInputStream(stream)
      val target = new File(dir, name)
      val fw = new FileWriter(target)
      src.foreach(fw.append)
      fw.close()
      src.close()
      target.getAbsolutePath
    }).last
  }



  override def train(c: Corpus): Unit = train(c, None)

  def train(c: Corpus, optMorpho: Option[Map[String, Set[String]]]): Unit = {

    val morpho = optMorpho.getOrElse(new Morfessor().buildMorphoMapping(c, None, lowerCase))
    val annotatedCorpus = c match {
      case r: CorpusRaw => Corpora.dummyTagger(r)
      case a: CorpusAnnotated => a
    }

    data = callPython(annotatedCorpus, morpho, nbDim, nOccMin)
    inVoc = data.keySet

  }

  override def completeVocabulary(targetVoc: Set[String]): Unit = {
    if(targetVoc.exists(!data.contains(_))) mimick(this, targetVoc)
  }

  override def apply(word: String): Array[Double] = data(word)

  def writeMorpho(morpho: Map[String, Set[String]], outFile: File): Unit = {
    val fw = new FileWriter(outFile)
    for((w,m) <- morpho) {
      fw.write(s"$w\t${m mkString(MORPH)}\n")
    }
    fw.close()
  }

  def callPython(c: CorpusAnnotated, morpho: Map[String, Set[String]], nDim: Int, nbOccMin: Int): Map[String, Array[Double]] = {
    val workdir = File.createTempFile("restaureTmp","msemb")
    workdir.delete()
    workdir.mkdir()
    val scriptPath = writePythonScriptToTemp(workdir)

    val corpusFile = new File(workdir, "corpus.conll")
    val morphoFile = new File(workdir, "morpho")


    Corpora.writeConll(c, corpusFile.getAbsolutePath)
    writeMorpho(morpho, morphoFile)

    //cmd...
    val args = Seq(
      corpusFile.getAbsolutePath,
      morphoFile.getAbsolutePath,
      nbOccMin.toString,
      nDim.toString,
      workdir.getAbsolutePath
    )

    process.Process.apply(Seq(python3, scriptPath) ++ args).!

    //read and return
    val resultSource = Source.fromFile(new File(workdir, s"${nDim}d-5n-lasti.vec")) // todo: negative sampling en paramètre
    val result = resultSource.getLines().drop(1).map { line =>
      val Array(w, v@_*) = line.split(" ")
      w -> v.map(_.toDouble).toArray
    } .toMap
    resultSource.close()
    result.updated("$$UNK$$",Array.fill(nbDim){0.0})
  }

}

object MorphoSyntacticEmbeddings {
  def initFromSettings(mimick: Mimick): MorphoSyntacticEmbeddings =
    new MorphoSyntacticEmbeddings(
      Settings.python3,
      Settings.embeddings.ndim,
      Settings.embeddings.nbOccMin,
      Settings.embeddings.ws,
      false, // todo: as option
      mimick)

  def loadFromSettings(): MorphoSyntacticEmbeddings = {
    val mimick = new Mimick(Settings.python2)
    val mse = initFromSettings(mimick)
    mse.fromFile(Settings.embeddings.file)
    mse
  }
}
