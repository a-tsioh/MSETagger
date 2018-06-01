package fr.magistry.nlp.mse


import java.io.File

import fr.magistry.nlp.datatypes._

import scala.sys.process._
/**
  * Allow to call an external command for tokenization
  * expect an input with one sentence per line and
  * an output on which split(" ") will correctly split tokens
  */
class ExternalTokenizer(cmd: String) {
  /**
    * call the tokenizer command of the instance
    * to the content of a file and create a raw corpus from the result
    * @param path
    * @return
    */
  def apply(path: String): CorpusRaw = {
    val results = (new File(path) #> cmd !!).split("\n")
    println(results.take(3) mkString "\n")
    CorpusRaw(
      path,
      results
        .filterNot(_.trim == "")
        .map(s =>
          SentenceRaw(
            s
              .replace("\u00a0"," ")
              .split(" ")
              .filterNot(_ == "")
              .map(TokenRaw)
          )
        )
    )
  }

}
