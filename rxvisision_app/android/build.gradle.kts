// --- ADD THESE IMPORTS AT THE VERY TOP ---
import com.android.build.gradle.BaseExtension
import org.gradle.api.Project
import org.gradle.api.Action

allprojects {
    repositories {
        google()
        mavenCentral()
    }

    // --- This forces every module to use the NEW Material library. ---
    configurations.all {
        resolutionStrategy {
            // Use 1.12.0 consistently
            force("com.google.android.material:material:1.12.0") 
        }
    }

    // --- THIS IS THE FIX ---
    // This hook runs for all projects (root and sub).
    // We will force the compileSdk here.
    afterEvaluate {
        if (project.plugins.hasPlugin("com.android.application") || project.plugins.hasPlugin("com.android.library")) {
            project.extensions.findByType(BaseExtension::class.java)?.apply {
                compileSdkVersion(36)
            }
        }
    }
}

val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

// --- COMBINED SUBPROJECTS BLOCK ---
subprojects {
    // Logic from your original file
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
    
    // Logic from your original file
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}