package fr.magistry.nlp

import java.io.File

package object utils {

  def rmdir(dir: File): Unit = {
    if(dir.isDirectory)
      dir.listFiles foreach rmdir
    dir.delete()
  }

}
