# Architecture

Two UML-style diagrams describe the toolkit: the class hierarchy and the flow of data
through the system. GitHub renders them directly from this file.

The main design rule is that **data ingestion is decoupled from analysis**. Providers only
retrieve and describe data. Analysis reads the cleaned tables and never knows which API
supplied them, so a new source or analysis does not change the core of the system.

## Class diagram

```mermaid
classDiagram
    direction LR

    class MarketDataDownloader {
        +providers
        +cache
        +from_json(paths)
        +download(tickers, lookback)
        +download_cob(cob, output)
    }
    class MarketDataProvider {
        <<abstract>>
        +name
        +kind
        +key_env
        +client
        +check_access()
    }
    class YahooProvider
    class AlphaVantageProvider
    class MarketDataAppProvider
    class MassiveProvider
    class FredProvider
    class ECBProvider
    class IBKRProvider
    class HttpClient
    class SeriesCache

    class RetrievedSeries {
        <<dataclass>>
        +data
        +metadata
        +warnings
    }
    class PriceSeries {
        <<dataclass>>
        +mean_return
        +std_return
        +update_statistics()
    }
    class ObservationSeries {
        <<dataclass>>
    }
    class DownloadResult {
        <<dataclass>>
        +summary
        +series
    }
    class CobResult {
        <<dataclass>>
        +folder
        +summary
        +reused
    }

    class ArchiveInventory {
        <<dataclass>>
        +requests
        +save(destination)
    }
    class ProcessedArchive {
        <<dataclass>>
        +cob
        +run_id
        +tables
        +summary
        +save(destination)
    }
    class PortfolioConfig {
        <<dataclass>>
        +reporting_currency
        +target_allocation
        +holdings
        +reference_data
    }

    class Portfolio {
        <<dataclass>>
        +processed
        +statistics
        +simulation
        +holdings
        +weights
        +from_saved(folder, modeling, monte_carlo)
        +monte_carlo(paths, horizon_sessions, assumptions)
        +plot_monte_carlo()
        +report(sections, limits)
        +plots_report(destination)
    }
    class PortfolioStatistics {
        <<dataclass>>
        +assets
        +portfolio
        +tables
        +report()
        +plots_report(destination)
        +save(destination)
    }
    class AssetStatistics {
        <<dataclass>>
        +instrument_id
        +frame
        +metrics
        +return_statistics
    }
    class MonteCarloResult {
        <<dataclass>>
        +settings
        +summary
        +tables
        +report()
        +plot(instrument_id, model)
        +save(destination)
    }

    MarketDataProvider <|-- YahooProvider
    MarketDataProvider <|-- AlphaVantageProvider
    MarketDataProvider <|-- MarketDataAppProvider
    MarketDataProvider <|-- MassiveProvider
    MarketDataProvider <|-- FredProvider
    MarketDataProvider <|-- ECBProvider
    MarketDataProvider <|-- IBKRProvider
    RetrievedSeries <|-- PriceSeries
    RetrievedSeries <|-- ObservationSeries

    MarketDataDownloader o-- "1..*" MarketDataProvider
    MarketDataDownloader o-- "0..1" SeriesCache
    MarketDataProvider --> HttpClient : uses
    YahooProvider ..> PriceSeries : creates
    AlphaVantageProvider ..> PriceSeries : creates
    MarketDataAppProvider ..> PriceSeries : creates
    MassiveProvider ..> PriceSeries : creates
    FredProvider ..> ObservationSeries : creates
    ECBProvider ..> ObservationSeries : creates
    MarketDataDownloader ..> DownloadResult : download()
    DownloadResult o-- "0..*" RetrievedSeries
    MarketDataDownloader ..> CobResult : download_cob()

    ArchiveInventory ..> ProcessedArchive : process_inventory()
    ProcessedArchive ..> PortfolioConfig : validated by
    Portfolio o-- "1" ProcessedArchive
    Portfolio *-- "1" PortfolioStatistics
    Portfolio *-- "0..1" MonteCarloResult
    PortfolioStatistics *-- "1..*" AssetStatistics
```

`IBKRProvider` has no `creates` arrow because it saves native responses to the archive
instead of building a series object. All sources meet again in the cleaned tables.

## System flow

```mermaid
flowchart TD
    CONFIG[/"config/*.json<br/>requests, portfolio, cleaning, modeling, Monte Carlo"/]
    RUN["run.py<br/>choose workflow"]
    APIS[("Yahoo, Alpha Vantage, MarketData,<br/>Massive, FRED, ECB, IBKR")]
    DOWNLOAD["MarketDataDownloader.download_cob()"]
    ARCHIVE[("Raw COB archive<br/>tests/data_dumps or data/demo_dumps")]
    INSPECT["inspect_archive()<br/>ArchiveInventory"]
    CLEAN["process_inventory()<br/>ProcessedArchive: selected prices and references"]
    PORTFOLIO["Portfolio"]
    STATS["PortfolioStatistics<br/>AssetStatistics per holding"]
    MC["Portfolio.monte_carlo()<br/>MonteCarloResult"]
    OUTPUT[/"processed/COB/run-id<br/>CSV tables, Markdown reports, PNG charts"/]

    CONFIG --> RUN
    RUN -- "2 Download or 3 Demo" --> DOWNLOAD
    DOWNLOAD <--> APIS
    DOWNLOAD --> ARCHIVE
    RUN -- "1 Process saved archive" --> ARCHIVE
    ARCHIVE --> INSPECT
    INSPECT --> CLEAN
    CLEAN --> PORTFOLIO
    PORTFOLIO -- "on creation" --> STATS
    PORTFOLIO -- "method call" --> MC
    STATS --> OUTPUT
    MC --> OUTPUT
    PORTFOLIO -- "report() and plots_report()" --> OUTPUT
```

## Extending the toolkit

| To add | Change | Core left untouched |
|---|---|---|
| A data source | A `MarketDataProvider` subclass, its entry in `PROVIDERS` and a JSON file | Cleaning, statistics, Monte Carlo |
| An instrument | The provider JSON files and `config/portfolio.json` | All code |
| A simulation setting | `config/monte_carlo.json` or a `Portfolio.monte_carlo()` argument | Retrieval and cleaning |
| A portfolio analysis | A `Portfolio` method reading its existing tables | Retrieval and cleaning |
