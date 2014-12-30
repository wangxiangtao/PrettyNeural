import AssemblyKeys._  // put this at the top of the file

assemblySettings

name := "PrettyMatch"

version := "1.0"

scalaVersion := "2.11.4"

autoScalaLibrary := true

resolvers ++= Seq(
  "Sonatype Nexus Releases" at "https://oss.sonatype.org/content/repositories/releases",
  "Maven Restlet" at "http://maven.restlet.org",
  "Clojars.org" at "http://clojars.org/repo"
)

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-library-all" % "2.11.4",
  "org.apache.solr" % "solr-core" % "4.10.1",
  "org.apache.solr" % "solr-solrj" % "4.10.1",
  "com.aliasi" % "lingpipe" % "4.0.1",
  "junit" % "junit" % "4.8.1",
  "commons-logging" % "commons-logging" % "1.1.1",
  "org.ahocorasick" % "ahocorasick" % "0.2.4",
  "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.11",
  "de.bwaldvogel" % "liblinear" % "1.94",
  "mysql" % "mysql-connector-java" % "5.1.24",
  "com.google.guava" % "guava" % "18.0",
  "org.apache.commons" % "commons-lang3" % "3.3.2"
)


