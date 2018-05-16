package fr.magistry.nlp.utils

import java.io.{File, FileWriter}

import scala.io.Source
import scala.sys.process
import fr.magistry.nlp.datatypes._
import fr.magistry.nlp.mse.Corpora


class Morfessor(args: Map[String,String]= Map.empty) {
  case class Affixation(form: String, root: String, sfx: String, afx:Set[String])

  val cmd = "morfessor" //Settings.tools.morfessor

  private val MORPH = "\ue00b"

  val config: Map[String,String] =
    Map(
      "d" -> "ones", // train on types
      "-output-format" -> "{compound}\\t{analysis}\\n"
    ) ++ args


  def buildAffixesSet(trainingCorpus: Corpus, words: Set[String]): Set[Affixation] = {
    val workDir = File.createTempFile("mseTmp", "Morfessor")
    workDir.delete()
    workDir.mkdir()
    val trainFile = new File(workDir.getAbsolutePath, "train.txt")
    val lexFile = new File(workDir.getAbsolutePath, "lex.txt")
    val outputFile = new File(workDir.getAbsolutePath, "model")

    //préparation des fichiers d'input
    val trainFW = new FileWriter(trainFile)
    (trainingCorpus match {
      case ca: CorpusAnnotated => ca.content.map {s => (s.tokens.map(_.form) mkString " ") + "\n"}
      case cr: CorpusRaw => cr.content.map {s => (s.tokens.map(_.form) mkString " ") + "\n"}
    }).foreach(trainFW.write)
    trainFW.close()

    val lexFW = new FileWriter(lexFile)
    words.foreach {w =>
      lexFW.write(w)
      lexFW.write('\n')
    }
    lexFW.close()

    //préparation de la liste d'arguments
    val args =  (config ++ Map(
      "-traindata" -> trainFile.getAbsolutePath,
      "T" -> lexFile.getAbsolutePath
    )).flatMap( {case (k,v) => Seq(s"-$k",v)})

    val outStream = process.Process.apply(Seq(cmd) ++ args).lineStream

    val result = outStream.map { line =>
      val Array(w, analysis) = line.stripLineEnd.split('\t')
      val morphemes = s"%%$analysis%%".split(' ')
      if (morphemes.length > 1) {
        val root = morphemes.take(morphemes.length - 1) mkString ""
        val sfx = morphemes.last
        Affixation(w, root, sfx, morphemes.toSet)// + ("$sow$" + morphemes.head) + (morphemes.last + "$eow$"))
      }
      else Affixation(w, w, "-0", Set.empty)
    }

    workDir.delete()

    result.toSet
  }

  def buildMorphoMapping(c:Corpus, words: Option[Set[String]], lower: Boolean=false): Map[String, Set[String]] = {
    val wordlist = words.getOrElse(Corpora.vocabularyOfCorpus(c, lower))
    buildAffixesSet(c,wordlist).foldLeft(Map.empty[String, Set[String]]) {
      (m, a) => m + (a.form -> a.afx)
    }
  }

  def writeMorphoMapping(targetFile: String, data: Map[String, Set[String]]): Unit =
    writeMorphoMapping(new File(targetFile), data)

  def writeMorphoMapping(targetFile: File, data: Map[String, Set[String]]): Unit = {
    val fw = new FileWriter(targetFile)
    data foreach {case  (w,m) => fw.write(s"$w\t") ; fw.write(m.mkString(MORPH)); fw.write('\n') }
    fw.close()
  }

  def readMorphoMapping(file: String): Map[String, Set[String]] = readMorphoMapping(new File(file))

  def readMorphoMapping(file: File): Map[String, Set[String]] = {
    val src = Source.fromFile(file)
    val mapping = src.getLines().map {l =>
      val Array(w,m) = l.split("\t",2)
      w -> m.split(MORPH).toSet
    }   .toMap
    src.close()
    mapping
  }


}
