plugins { id("com.android.application") }
android {
    namespace = "org.adaptiveslm.demo"
    compileSdk = 36
    defaultConfig {
        applicationId = "org.adaptiveslm.demo"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0-harness"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
