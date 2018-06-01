package fr.magistry.nlp.utils

import java.io.{File, FileWriter}

import fr.magistry.nlp.mse.Embeddings

import scala.io.Source
import scala.sys.process

class Mimick(python2: String,
             mem: String = "512",
             useGPU: Boolean = false) {

  private def writePythonScriptToTemp(dir: File): Array[String] = {
    assert(dir.isDirectory)
    Array("model.py", "make_dataset.py") map { name =>
      val stream = this.getClass.getResourceAsStream("/python/mimick/" + name)
      val src = Source.fromInputStream(stream)
      val target = new File(dir, name)
      val fw = new FileWriter(target)
      src.foreach(fw.append)
      fw.close()
      src.close()
      target.getAbsolutePath
    }
  }

  /**
    * train a mimick model and complete a list vector by adding vectors for OOV
    *
    * @param embeddings embeddings for known words
    * @param voc target vocabulary
    * @return vectors for target vocabulary
    */
  def apply(embeddings: Embeddings, voc: Set[String]): Unit = {
    val workDir = File.createTempFile("restaureTmp","Mimick")
    workDir.delete()
    workDir.mkdir()
    val Array(modelScript, makeDatasetScript) = writePythonScriptToTemp(workDir)

    val vocabFile = new File(workDir,"vocab")
    val vocabFW = new FileWriter(vocabFile)
    voc.foreach(w => vocabFW.write(s"$w\n"))
    vocabFW.close()

    val vectorsFile = new File(workDir, "vectors")
    embeddings.toFile(vectorsFile)

    val dataFile = new File(workDir, "model")
    val outputFile = new File(workDir, "output")

    val cmd1 = Seq(
      python2,
      makeDatasetScript,
      "--vectors", vectorsFile.getAbsolutePath,
      "--output", dataFile.getAbsolutePath,
      "--vocab", vocabFile.getAbsolutePath,
      "--w2v-format"
    )
    process.Process.apply(cmd1).!

    val cmd2 = Seq(
      python2,
      modelScript,
      "--dataset", dataFile.getAbsolutePath,
      "--vocab", vocabFile.getAbsolutePath,
      "--output", outputFile.getAbsolutePath,
      "--dynet-mem", mem
    ) ++ (if (useGPU) Seq("--dynet-gpu") else Seq())

    process.Process.apply(cmd2).!
    embeddings.fromFile(outputFile.getAbsoluteFile)
  }

  /*
  def train(embeddings: Embeddings): Unit = {
    val workDir = File.createTempFile("restaureTmp","Mimick")
    workDir.delete()
    workDir.mkdir()
    val Array(modelScript, makeDatasetScript) = writePythonScriptToTemp(workDir)

    //val vocabFile = new File(workDir,"vocab")
    //val vocabFW = new FileWriter(vocabFile)
    //voc.foreach(w => vocabFW.write(s"$w\n"))
    //vocabFW.close()

    val vectorsFile = new File(workDir, "vectors")
    embeddings.toFile(vectorsFile)

    val dataFile = new File(workDir, "model")
    val outputFile = new File(workDir, "output")

    val cmd1 = Seq(
      python2,
      makeDatasetScript,
      "--vectors", vectorsFile.getAbsolutePath,
      "--output", dataFile.getAbsolutePath,
      "--vocab", vocabFile.getAbsolutePath,
      "--w2v-format"
    )
    process.Process.apply(cmd1).!

  } */

}

