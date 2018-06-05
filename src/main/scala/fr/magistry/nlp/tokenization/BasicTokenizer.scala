package fr.magistry.nlp.tokenization

import fr.magistry.nlp.datatypes.{CorpusRaw, SentenceRaw, TokenRaw}

object BasicTokenizer {
  val punctRE = "(?! )([\\.,!?;:\\(\\)\"]+)".r("punct")
  val punctRE2 = "([\\.,!?;:\\(\\)'’\"]+)(?! )".r("punct")

  def fromFile(path: String): CorpusRaw = {
    val data = scala.io.Source.fromFile(path)

    val sentences = data.getLines().flatMap {line =>
      val tokens =
        ( {s:String => s.replace('\t', ' ').replace("\u00a0"," ")}
          andThen {punctRE.replaceAllIn(_:String, " $1")}
          andThen {punctRE2.replaceAllIn(_:String,"$1 ")}
          andThen {_.split(" ") map {_.trim} filter {_.nonEmpty} }
          )(line)

      def aux(toks: Stream[String], acc:List[List[TokenRaw]]): List[SentenceRaw] = {
        toks match {
          case Stream.Empty => acc.reverseMap(toks => SentenceRaw(toks.reverse.toArray))
          case tok #:: tail if Seq(".", "!", "?").contains(tok) =>
            aux(tail, Nil :: (TokenRaw(tok) :: acc.head) :: acc.tail)
          case tok #:: tail =>
            aux(tail, ( TokenRaw(tok) :: acc.head) :: acc.tail)
        }
      }
      aux(tokens.toStream, List(Nil)).filter(_.tokens.nonEmpty)
    }
    val c = CorpusRaw(
      path,
      sentences.take(100).toArray
    )
    data.close()
    c
  }
}
