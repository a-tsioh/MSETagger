name := "MSETagger"

version := "0.1"

scalaVersion := "2.11.6"

libraryDependencies ++= Seq(
//  "org.apache.commons" % "commons-math3" % "3.6.1",
//  "org.apache.commons" % "commons-text" % "1.2",
  "com.typesafe" % "config" % "1.3.1",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0",
  "org.apache.commons" % "commons-io" % "1.3.2"
//  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models-german",
//  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models-french",
//  "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models",
//  "org.scalanlp" % "breeze_2.11" % "0.13.1",
//  "org.scalanlp" % "breeze-viz_2.11" % "0.13.1",
//  "org.scalanlp" % "breeze-natives_2.11" % "0.13.1"
)