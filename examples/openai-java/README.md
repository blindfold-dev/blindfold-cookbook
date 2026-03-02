# OpenAI + Blindfold (Java)

Protect PII in OpenAI chat conversations. User messages are tokenized before reaching OpenAI, and responses are detokenized to restore original values.

## How it works

1. **Tokenize** user message — `John Smith` becomes `<Person_1>`
2. **Send** tokenized text to OpenAI — no real PII leaves your system
3. **Detokenize** AI response — `<Person_1>` becomes `John Smith` again

## Prerequisites

- Java 11+
- Maven or Gradle

## Setup

Add dependencies to your `pom.xml`:

```xml
<dependencies>
    <dependency>
        <groupId>dev.blindfold</groupId>
        <artifactId>blindfold-sdk</artifactId>
        <version>1.0.0</version>
    </dependency>
    <dependency>
        <groupId>com.openai</groupId>
        <artifactId>openai-java</artifactId>
        <version>0.8.0</version>
    </dependency>
    <dependency>
        <groupId>io.github.cdimascio</groupId>
        <artifactId>dotenv-java</artifactId>
        <version>3.0.0</version>
    </dependency>
</dependencies>
```

Create a `.env` file:

```bash
cp ../../../.env.example .env
# Edit .env with your API keys
```

## Run

```bash
mvn compile exec:java -Dexec.mainClass="Main"
```

## Offline mode

Works without a Blindfold API key. Omit `BLINDFOLD_API_KEY` from `.env`
and PII detection runs locally using built-in regex patterns.
