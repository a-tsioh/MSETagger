package fr.magistry.nlp

import fr.magistry.nlp.datatypes.{CorpusAnnotated, GoldCorpus}
import fr.magistry.nlp.mse.Corpora

class Eval(val gold: GoldCorpus,val test: CorpusAnnotated) {

  private def computeStats() = {
    println(gold.content.length)
    println(test.content.length)
    Corpora.writeConll(test,"/tmp/debug.conll")
    assert(gold.content.length == test.content.length)

    var good = 0
    var total = 0
    val confusionMatrix = collection.mutable.HashMap.empty[(String,String),Int].withDefaultValue(0)
    for((goldSentence, testSentence) <- gold.content.zip(test.content)) {
      assert(goldSentence.tokens.length == testSentence.tokens.length)
      for((goldTok, testTok) <- goldSentence.tokens.zip(testSentence.tokens)) {
        total += 1
        if(goldTok.pos == testTok.pos) {
          good += 1
        }
        confusionMatrix((goldTok.pos, testTok.pos)) += 1
      }
    }
    (good, total, confusionMatrix.toMap.withDefaultValue(0))
  }

  val (good, total, confusionMatrix) = computeStats()


  def accuracy: Double = good.toDouble / total.toDouble


  def confusion: String = {
    val tagsetG = Corpora.tagsetOfCorpus(gold)
    val tagsetT = Corpora.tagsetOfCorpus(test)

    val lines = (for(tt <- tagsetT) yield {
      tt + " & "  + (tagsetG.map(confusionMatrix(_,tt).toString) mkString " & ")
    }) mkString " \\\\ \n"

    " & " + (tagsetG mkString " & ") +"\\\\ \n" + lines
  }

}
