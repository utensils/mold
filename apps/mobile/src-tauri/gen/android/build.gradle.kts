buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.11.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.25")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// The Tauri scanner defaults to the Play Services artifact, whose decoder is
// downloaded after first launch. Pairing must work on a fresh or offline
// device, so compile that plugin against Google's API-compatible bundled model.
project(":tauri-plugin-barcode-scanner") {
    afterEvaluate {
        val implementationDependencies = configurations
            .getByName("implementation")
            .dependencies
        implementationDependencies
            .filter {
                it.group == "com.google.android.gms" &&
                    it.name == "play-services-mlkit-barcode-scanning"
            }
            .forEach {
                implementationDependencies.remove(it)
            }
        dependencies.add(
            "implementation",
            "com.google.mlkit:barcode-scanning:17.3.0",
        )
    }
}

tasks.register("clean").configure {
    delete("build")
}
