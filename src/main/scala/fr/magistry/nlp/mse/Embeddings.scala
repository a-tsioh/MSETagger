package fr.magistry.nlp.mse

import scala.io.Source
import java.io.{File, FileWriter}

import fr.magistry.nlp.datatypes._



abstract class Embeddings(nbDim: Int, nOccMin: Int, ws: Int, lowerCase: Boolean) {

  var data: Map[String, Array[Double]] = Map.empty
  var inVoc: Set[String] = Set.empty

  def train(c: Corpus): Unit

  def completeVocabulary(targetVoc: Set[String]): Unit

  def apply(word: String): Array[Double]

  def toFile(file: File): Unit = {
    val fw = new FileWriter(file)
    fw.write(s"${data.size} $nbDim\n")
    for ((w, v) <- data) {
      fw.write(s"$w ${v.map(_.toString) mkString " "}\n")
    }
    fw.close()
  }

  def inVocToFile(file: File): Unit = {
    val fw = new FileWriter(file)
    fw.write(inVoc mkString "\n")
    fw.close()
  }

  def fromFile(file: File): Unit = {
    val src = Source.fromFile(file)
    data = Map("$$UNK$$" -> Array.fill(nbDim){0.0}) ++ src.getLines().drop(1).map { line =>
      val Array(w, v@_*) = line.split(" ")
      w -> v.map(_.toDouble).toArray
    } // .toMap
    src.close()
  }

}