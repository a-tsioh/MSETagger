package fr.magistry.nlp.mse

import java.io.FileWriter

import fr.magistry.nlp.datatypes._

import scala.io.Source

object Corpora {

  def loadConllCorpus(path: String): CorpusAnnotated = {
    def parseOneLine(line: String): TokenWithPOS = {
      val Array(_, form, lemma, tag1, tag2, _*) = line.trim.split("\t")
      TokenWithPOS(form.replace(" ","_").replace('\u2009', '_'), tag1)
    }

    val src = Source.fromFile(path)
    val dataStream = src.getLines().toStream

    def consumeStream(stream: Stream[String], acc: List[String]): Stream[List[String]] = {
          stream match {
            case Stream.Empty => Stream.Empty
            case line #:: tail if line.trim.startsWith("#") => consumeStream(tail, acc)
            case line #:: tail if line.trim.isEmpty => acc.reverse #:: consumeStream(tail, Nil)
            case line #:: tail =>
              consumeStream(tail, line :: acc)
          }
        }

    CorpusAnnotated(
      path,
      consumeStream(dataStream, Nil).par.map { sentenceData =>
        SentenceAnnotated(sentenceData.map(parseOneLine).toArray)
      } .toArray)
  }

  /**
    * load conll file and add the trait `Gold` to the resulting CorpusAnnotated
    * @param path
    * @return
    */
  def loadGoldCorpus(path: String): GoldCorpus = loadConllCorpus(path).asInstanceOf[GoldCorpus]

  /**
    * add the same tag to all tokens to create an annotated corpus
    * @param raw
    * @param tag
    * @return
    */
  def dummyTagger(raw: CorpusRaw, tag: String="X"): CorpusAnnotated =
    CorpusAnnotated(
      raw.name,
      raw.content.map {s =>
        SentenceAnnotated(s.tokens.map {t =>
          TokenWithPOS(t.form, tag)
        })
      }
    )

  def vocabularyOfCorpus(c: Corpus, lower: Boolean=false): Set[String] = {
    val wordlist = c match {
      case r: CorpusRaw => r.content.foldLeft(Set.empty[String]) {case (set, sent) => set.union(sent.tokens.map(_.form).toSet)}
      case a: CorpusAnnotated => a.content.foldLeft(Set.empty[String]) {case (set, sent) => set.union(sent.tokens.map(_.form).toSet)}
    }
    if(lower) wordlist.map(_.toLowerCase)
    else wordlist
  }

  def writeConll(c: CorpusAnnotated, path: String) = {
    val fw = new FileWriter(path)
    c
      .content
      .par
      .map {s => s.tokens.zipWithIndex.map {case (t,i)  => s"$i\t${t.form}\t_\t${t.pos}" } mkString "\n" }
      .seq
      .foreach {s => fw.append(s) ; fw.append("\n\n") }
    fw.close()
  }
}
