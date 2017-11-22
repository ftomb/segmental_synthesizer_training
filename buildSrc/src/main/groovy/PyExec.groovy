import org.gradle.api.DefaultTask
import org.gradle.api.tasks.*

class PyExec extends DefaultTask {

    @InputFile
    File pyFile

    @Input
    Object args = []

    @Optional
    @InputDirectory
    File srcDir

    @Optional
    @OutputDirectory
    File destDir

    @TaskAction
    void run() {
        project.exec {
            commandLine = ['python3', pyFile] + this.args
        }
    }
}
