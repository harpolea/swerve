pipeline {
	agent any
	stages {
		stage('build') {
			steps {
				sh "make test"
				sh "cd testing"
				sh "./unit_tests"
			}
		}
	}
}
