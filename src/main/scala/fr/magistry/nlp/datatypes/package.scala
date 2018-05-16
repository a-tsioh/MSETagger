package fr.magistry.nlp

package object datatypes {
  abstract sealed class Corpus(name: String)
  abstract sealed class Sentence
  abstract sealed class Token

  trait Gold
  trait Retagged

  case class TokenRaw(form: String) extends Token
  case class TokenWithPOS(form: String, pos: String) extends Token

  case class SentenceRaw(tokens: Array[TokenRaw]) extends Sentence
  case class SentenceAnnotated(tokens: Array[TokenWithPOS]) extends Sentence

  case class CorpusRaw(name: String, content: Array[SentenceRaw]) extends Corpus(name)
  case class CorpusAnnotated(name: String, content: Array[SentenceAnnotated]) extends Corpus(name)

  type GoldCorpus = CorpusAnnotated with Gold
  type RetaggedCorpus = CorpusAnnotated with Retagged


  object Options {
    abstract sealed class CorpusLoadingFashion
    case object ReadConll extends CorpusLoadingFashion
    case object ExternalTokenizer extends CorpusLoadingFashion
   // case object CallOpenNLPTokenizer extends CorpusLoadingFashion // TODO: specify model (how ?)
  }
}
