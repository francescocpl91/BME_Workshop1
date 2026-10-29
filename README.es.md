# PythonToolBox_Taller1Master - Datos de mercado
[← Volver al README principal](README.md)

Los archivos de entrada se encuentran en `config/*.json`. La configuración operativa y los nombres de las variables de entorno utilizadas para las credenciales están definidos en `run.py`. Los módulos de la biblioteca se encuentran directamente dentro de `src/`.

## Elegir un flujo de trabajo

Ejecuta `run.py` con el intérprete de Python del proyecto. El programa te pedirá que elijas una de estas opciones:

- **1 - Procesar un archivo guardado:** introduce una fecha COB ya guardada. El programa inventaría, limpia y combina los datos locales sin conectarse a las API ni solicitar una fecha de inicio del historial. Después genera las estadísticas y los escenarios históricos de Monte Carlo configurados.
- **2 - Descargar y procesar:** introduce la fecha COB y la fecha de inicio del historial. El programa recupera o reutiliza un archivo compatible y, a continuación, inventaría, limpia, combina, analiza y simula los datos automáticamente.

Ambas opciones utilizan el mismo proceso de tratamiento de datos. También funcionan los alias `process`, `inventory` y `download`.

La función independiente de la biblioteca `inspect_archive()` sigue realizando únicamente el inventario, mientras que `download_cob()` se encarga exclusivamente de la recuperación de datos.

El punto de entrada del programa conecta estas etapas sin solicitar una fecha adicional.

Se rechazan los archivos inexistentes o incompletos, así como aquellos que tengan un bloqueo `.running` activo.

## Estadísticas históricas

Los dos flujos de trabajo de `run.py` calculan estadísticas después de la limpieza de los datos.

La configuración se encuentra en `config/modeling.json`, mientras que los identificadores de los activos (*tickers*) y sus pesos permanecen en `config/portfolio.json`.

Esta etapa funciona sin conexión a internet y analiza únicamente las posiciones configuradas en la cartera.

Las opciones y los instrumentos de referencia no forman parte de los pesos de la cartera.

Las estadísticas tienen carácter descriptivo. La etapa independiente de Monte Carlo, explicada más adelante, se ejecuta después.

No se realiza valoración de opciones.

El archivo `src/statistics.py` proporciona las clases `AssetStatistics` (estadísticas muestrales automáticas) y `PortfolioStatistics` (con los métodos `report()`, `plots_report()` y `save()`).

Estos objetos analíticos trabajan con las observaciones ya limpiadas, pero no sustituyen a los objetos `PriceSeries` utilizados para recuperar los datos.

La opción `statistics.return_types` puede configurarse como `["simple", "log"]`, que es el valor predeterminado, o seleccionarse únicamente uno de los dos tipos de rentabilidad.

Un único motor de cálculo genera ambos análisis:

- **Rentabilidad simple:** `simple = P_t/P_previous - 1`
- **Rentabilidad logarítmica:** `log = log1p(simple)`

Ambos cálculos utilizan exactamente los mismos intervalos válidos.

Para cada tipo de rentabilidad se obtienen:

- Media y desviación estándar diarias.
- Media y volatilidad anualizadas.
- Asimetría y exceso de curtosis.
- Cuantiles empíricos y valores extremos.
- Volatilidad móvil.
- Ajustes a distribuciones normal y t de Student.
- Covarianza y correlación.
- Comparaciones durante periodos de estrés financiero.

Esta etapa descriptiva no genera simulaciones.

Todas las filas de los archivos CSV que dependen de la rentabilidad incluyen el campo `return_type`.

La media anualizada se calcula multiplicando la media diaria por `periods_per_year`:

- Para rentabilidades simples, representa la media aritmética anualizada.
- Para rentabilidades logarítmicas, representa el crecimiento logarítmico aditivo anualizado.

**No equivale a la tasa de crecimiento anual compuesta (CAGR).**

Por defecto, la volatilidad utiliza la desviación estándar muestral (`ddof=1`) multiplicada por `sqrt(252)`.

El exceso de curtosis utiliza el valor cero como referencia para una distribución normal.

Los valores que no pueden calcularse se muestran vacíos o como `null`, nunca como cero.

Los gráficos representan las rentabilidades simples en porcentaje y las rentabilidades logarítmicas multiplicadas por 100, indicando expresamente que se trata de unidades de rentabilidad logarítmica.

La riqueza normalizada, la rentabilidad acumulada, la CAGR, la caída porcentual máxima (*drawdown*) y la cobertura temporal se calculan una sola vez a partir de los precios o de la evolución del patrimonio, y se comparten entre ambos tipos de rentabilidad.

La CAGR utiliza los años naturales transcurridos.

El informe muestra las fechas reales junto a estas métricas. Los historiales individuales de algunos activos pueden comenzar antes que la primera fecha común disponible para toda la cartera.

Las rentabilidades logarítmicas de la cartera se calculan a partir de la evolución del patrimonio de la cartera ya construida. **Nunca se calculan como una media ponderada de las rentabilidades logarítmicas de sus componentes.**

Para utilizar directamente la biblioteca, `asset.return_statistics["simple"]` y `asset.return_statistics["log"]` contienen las métricas, los análisis de estrés, los cálculos móviles y los resultados de las distribuciones.

Por su parte, `asset.path_metrics` y `asset.stress_paths` contienen los resultados compartidos relativos a la evolución del patrimonio.

Se mantienen los atributos originales de acceso rápido `metrics`, `stress`, `rolling` y `distribution`. Cuando está habilitado el análisis de rentabilidad simple, se utiliza preferentemente este; en caso contrario, se muestra el análisis logarítmico identificado como tal.

Para evitar ambigüedades, se recomienda acceder mediante `return_statistics[kind]`.

### Construcción y análisis de la cartera

Las estadísticas de la cartera representan una simulación histórica hipotética de una estrategia de **comprar y mantener (*buy and hold*)**.

Los pesos iniciales actuales se aplican una sola vez, en la primera fecha para la que existe un precio observado común a todos los activos. A partir de ese momento, los pesos de las distintas posiciones evolucionan según sus rentabilidades.

Por tanto, **no se trata de una cartera que se rebalancea diariamente para mantener una distribución del 50 % y 50 %**.

Las series de precios ajustados ya incorporan los ajustes correspondientes a las distribuciones, por lo que los dividendos en efectivo no se añaden una segunda vez.

Las carteras construidas únicamente con precios sin ajustar se identifican como `price_return`.

Todas las posiciones activas deben estar disponibles en una misma moneda y utilizar criterios de rentabilidad compatibles.

Si faltan datos de alguna posición, esta no se elimina ni se redistribuye su peso entre los demás activos.

Cuando se mezclan monedas o criterios de rentabilidad incompatibles, no es posible calcular la evolución conjunta del patrimonio de la cartera. Sin embargo, las estadísticas individuales de los activos siguen estando disponibles.

Los historiales denominados en dólares estadounidenses (USD) producen estadísticas de cartera identificadas en USD, incluso cuando la configuración solicita presentar los resultados en euros (EUR).

**No se realiza ninguna conversión de divisas ni se supone que el tipo de cambio permanece constante.**

### Tratamiento de los datos ausentes

Las sesiones de mercado sin observaciones nunca se rellenan artificialmente.

Una rentabilidad diaria no puede calcularse atravesando un intervalo en el que faltan datos.

La correlación y la covarianza se calculan utilizando únicamente observaciones completas cuyos intervalos tengan exactamente las mismas fechas de inicio y fin.

Las estadísticas móviles conservan las fechas sin datos, de modo que esos huecos interrumpen las ventanas de cálculo.

No se presentan la rentabilidad acumulada, la CAGR ni la caída máxima (*maximum drawdown*) cuando la trayectoria observada contiene precios ausentes o intervalos excluidos deliberadamente.

La ausencia de datos anterior a la primera observación de un instrumento no se considera un hueco dentro de su historial.

### Comparaciones durante periodos de crisis

Las ventanas de estrés correspondientes a la crisis financiera global (GFC), la pandemia de COVID-19 y el año 2022 son periodos ilustrativos que pueden configurarse.

Para cada periodo se exportan:

- La cobertura temporal realmente disponible.
- El número de observaciones.
- La volatilidad.
- La caída máxima (*drawdown*).
- La correlación entre activos.

Se excluye cualquier rentabilidad cuyo intervalo comience antes del periodo de estrés seleccionado.

Las estadísticas del historial completo utilizan las fechas disponibles de cada activo por separado.

El análisis de dependencia entre activos utiliza únicamente la muestra común, cuyo número de observaciones se informa de manera independiente.

### Ajustes de distribuciones estadísticas

Los ajustes de densidad a las distribuciones normal y t de Student tienen carácter descriptivo y **no constituyen los parámetros de las simulaciones**.

El parámetro de escala de la distribución t de Student no equivale a su desviación estándar.

Los ajustes cuya varianza no está definida se identifican expresamente.

Cuando los historiales son demasiado cortos o presentan valores constantes, no se realiza el ajuste.

Los errores al ajustar una distribución se registran sin impedir el cálculo de las demás estadísticas.

### Archivos generados por las estadísticas

Los resultados se guardan en la siguiente estructura de carpetas:

`processed/<COB>/<processing-id>/statistics/<statistics-id>/`

Los archivos generados son:

- `statistics.csv`: momentos estadísticos de las rentabilidades de cada serie, identificados como simples o logarítmicas.
- `path_statistics.csv`: crecimiento, CAGR, caídas máximas y cobertura temporal de cada serie.
- `daily_statistics.csv`: precios seleccionados, ambos tipos de rentabilidad, fechas de inicio de los intervalos, patrimonio normalizado y caídas. Cada fecha aparece una sola vez por serie.
- `rolling_volatility.csv`, `dependence.csv` y `rolling_correlations.csv`: diagnósticos estadísticos identificados por tipo de rentabilidad.
- `aligned_returns.csv`: intervalos comunes para cada tipo de rentabilidad, con columnas `return::<id>`.
- `stress_statistics.csv` y `stress_dependence.csv`: comparaciones de rentabilidades durante periodos de crisis.
- `stress_path_statistics.csv`: crecimiento, caídas y cobertura real durante las ventanas de estrés.
- `distributions.csv`: parámetros de las distribuciones normal y t de Student, junto con el estado de sus ajustes.
- `report.md` y `manifest.json`: hipótesis utilizadas, advertencias, identificación de los datos de entrada y huellas digitales (*hashes*) de los archivos.
- `plots/simple/` y `plots/log/`: gráficos de volatilidad, histogramas, diagnósticos Q-Q de cada serie y comparaciones de correlación según el tipo de rentabilidad seleccionado.
- `plots/wealth.png` y `plots/drawdown.png`: gráficos compartidos de la evolución del patrimonio y sus caídas.

El esquema de salida de las estadísticas es ahora la **versión 2**.

Las métricas relativas a la evolución del patrimonio se han trasladado desde `statistics.csv` a `path_statistics.csv`, así como a su tabla equivalente para los periodos de estrés.

Las tablas de rentabilidades incorporan ahora el campo `return_type`.

Los informes antiguos permanecen sin cambios en sus carpetas originales.

El campo `aligned_observations` cuenta cada intervalo común una sola vez, sin duplicarlo por utilizar ambos tipos de rentabilidad.

El archivo `manifest.json` también registra `aligned_observations_by_return_type`.

Las claves de los gráficos en el manifiesto utilizan rutas relativas. Así, los gráficos de rentabilidades simples y logarítmicas que tienen el mismo nombre conservan cada uno su propia suma de verificación (*checksum*).

La identidad de cada ejecución estadística depende de los datos de entrada ya limpiados, los pesos de la cartera, la configuración, el código y las versiones de las dependencias.

Se conservan los archivos originales y los archivos limpiados.

**Que un análisis figure como completado significa que se ha exportado el informe, no necesariamente que estén disponibles todos los activos o todos los datos necesarios para el modelo.**

### Analizar directamente una ejecución ya procesada

Para analizar una ejecución cuyos datos ya se han limpiado, sin volver a examinar los archivos originales:

```python
from pathlib import Path
from src import analyze_saved_processing

folder = Path("processed/2026-10-08/<processing-id>")
analysis = analyze_saved_processing(folder, "config/modeling.json")
print(analysis.report())
analysis.save(folder / "statistics" / analysis.run_id)
```

Es necesario seleccionar expresamente el identificador de procesamiento (`processing-id`) deseado.

La biblioteca no intenta adivinar qué ejecución anterior ni qué configuración de cartera debe utilizar.

Antes del análisis se comprueba la suma de verificación del archivo de precios guardado.

Dentro de Python, `analyze_processed(result, config_path)` también acepta directamente el resultado almacenado en memoria que devuelven `process_archive` o `process_inventory`.

## Monte Carlo histórico

Los dos flujos de trabajo de `run.py` ejecutan `src/monte_carlo.py` después de calcular las estadísticas.

Si se establece `enabled=false` en `config/monte_carlo.json`, se guarda un estado que indica que la simulación está desactivada.

La recuperación de datos, su limpieza y las estadísticas descriptivas no dependen de que Monte Carlo termine correctamente.

Si el historial es insuficiente o faltan factores de conversión de divisas, se genera un informe que indica que la simulación no está disponible. En cambio, una configuración incorrecta o unos datos guardados corruptos provocan errores explícitos.

### Configuración inicial y fechas

La configuración inicial utiliza:

- **10.000 trayectorias por escenario.**
- **Patrimonio inicial normalizado de 100.**
- **Fecha de inicio:** 9 de octubre de 2026.
- **Fecha de finalización:** 17 de diciembre de 2027.
- **Calendario bursátil:** `XNYS`, correspondiente a la Bolsa de Nueva York.

La primera fecha representa el valor inicial de la inversión. La primera rentabilidad simulada se produce al finalizar la siguiente sesión bursátil.

Estos valores representan escenarios hipotéticos de inversión normalizada, no precios realmente observados el 9 de octubre.

La fecha de inicio debe corresponder a una sesión bursátil estrictamente posterior a la fecha COB del archivo utilizado.

Si la fecha de finalización cae en un festivo o fin de semana, se utiliza la última sesión bursátil anterior y se muestra una advertencia.

Las fechas se eligen expresamente en el archivo JSON. Ni el vencimiento de una opción ni el símbolo de un instrumento determinan el horizonte de simulación.

### Calibración con datos históricos

La calibración utiliza los últimos diez años naturales hasta la fecha COB, salvo que se indique otra fecha de inicio mediante `calibration.start`.

La media diaria de las rentabilidades logarítmicas de cada posición y la matriz conjunta de covarianzas muestrales se calculan utilizando exactamente los mismos intervalos válidos entre sesiones consecutivas.

Por defecto, se necesitan al menos **252 observaciones comunes**.

No se rellenan las observaciones ausentes, no se combinan intervalos de distinta duración y se excluyen los datos posteriores a la fecha COB.

El informe muestra las fechas reales y el número de observaciones utilizadas.

Esta primera versión del motor exige que todas las posiciones sigan el calendario bursátil configurado. Si se mezclan calendarios incompatibles, la simulación se declara no disponible.

### Cálculo de las rentabilidades simuladas

Para cada incremento logarítmico diario `g`, se utilizan las siguientes relaciones:

```text
g = media_logarítmica_histórica + perturbación_correlacionada

patrimonio_normalizado_siguiente =
    patrimonio_normalizado_anterior * exp(g)
```

La media logarítmica histórica ya incorpora la convención de crecimiento logarítmico. Por tanto, **no debe restarse otra vez la mitad de la varianza**.

En el modelo gaussiano, el crecimiento bruto esperado para un único periodo es:

```text
exp(media_logarítmica + varianza_logarítmica / 2)
```

El error estándar comunicado para la media diaria se calcula como:

```text
desviación_estándar_logarítmica / sqrt(número_de_observaciones)
```

Este cálculo supone que las observaciones son independientes y no recoge la incertidumbre asociada a posibles cambios de régimen de mercado.

La tendencia media y la volatilidad permanecen constantes durante todo el horizonte de previsión.

Las perturbaciones aleatorias son independientes entre sesiones.

### Simulación conjunta de la cartera

Los activos se simulan conjuntamente, en lugar de reducir toda la cartera a una única serie histórica previamente calculada.

Los pesos iniciales se aplican una sola vez al comenzar la simulación y después evolucionan según el comportamiento de cada activo.

El patrimonio total de la cartera se calcula como la suma ponderada de los patrimonios normalizados de sus posiciones.

**Los pesos no se restablecen diariamente.**

Los historiales de precios ajustados representan un patrimonio que ya incorpora la reinversión correspondiente. Por tanto, no se vuelven a añadir los dividendos.

Las posiciones cuyos datos faltan nunca se eliminan ni se redistribuyen entre las restantes.

Los pesos cuya suma se acepta dentro de la tolerancia de redondeo existente de `1e-8` se ajustan a una asignación total de uno exclusivamente para Monte Carlo.

Los informes conservan tanto los pesos configurados como los pesos efectivos y muestran una advertencia cuando se realiza este ajuste.

La configuración original de la cartera no se modifica.

Los errores de asignación mayores siguen siendo rechazados.

Las ganancias de la cartera se agregan tomando como referencia el patrimonio inicial. Así se evita que posiciones constantes generen pérdidas ficticias por errores de redondeo al sumar los valores ponderados.

La covarianza después de la conversión de divisas se calcula a partir del factor de simulación transformado. Esto evita varianzas negativas debidas a errores de precisión numérica cuando las perturbaciones de un activo y de su tipo de cambio casi se compensan.

No obstante, las matrices de covarianza que sean realmente inválidas siguen siendo rechazadas.

Todas las posiciones deben utilizar criterios compatibles de rentabilidad total (*total return*) o rentabilidad por precio (*price return*).

### Modelos matemáticos disponibles

Se ofrecen dos familias de modelos:

- **Gaussiano:** genera perturbaciones logarítmicas normales y correlacionadas, utilizando la covarianza histórica.
- **t de Student multivariante truncada:** ajusta conjuntamente un único parámetro de grados de libertad (`df`) mediante máxima verosimilitud con límites, manteniendo fijas la media y la covarianza históricas.

Los grados de libertad del modelo multivariante no se copian de los ajustes individuales de distribuciones t de Student que aparecen en el informe estadístico.

Una perturbación radial compartida relaciona las magnitudes de los movimientos simultáneos de los activos.

Una perturbación logarítmica t de Student sin restricciones no tiene un momento exponencial finito. Por tanto, el patrimonio esperado que implicaría ese modelo tampoco sería finito.

Por esta razón, el escenario t de Student implementado condiciona las perturbaciones a un radio finito y después reescala su covarianza condicional para recuperar la misma covarianza histórica que utiliza el modelo gaussiano.

El modelo se identifica expresamente como **truncado** en todos los resultados.

Los precios no se recortan después de realizar la simulación.

### Configuración de la t de Student

El esquema de entrada y salida de Monte Carlo es ahora la **versión 2**.

La configuración del modelo t de Student es:

```json
"student_t": {
  "fit_df": true,
  "df_bounds": [2.1, 50],
  "excluded_tail_probability": 0.0005,
  "sensitivity_tail_probabilities": [0.002, 0.00005]
}
```

El campo `fit_df` debe permanecer en `true`.

Se rechazan los valores fijos de grados de libertad (`df`) y los radios de truncamiento fijados manualmente.

Los límites `df_bounds` son restricciones explícitas del optimizador, no resultados del ajuste.

Si el ajuste termina en uno de esos límites, se informa de ello. El programa no inventa silenciosamente unos grados de libertad alternativos ni amplía automáticamente los límites.

Cuando cambia la ventana histórica o el conjunto de activos, se vuelven a calibrar las medias, la covarianza y los grados de libertad conjuntos.

Un cambio exclusivamente en los pesos de la cartera no modifica esos parámetros si los factores utilizados y la muestra histórica siguen siendo idénticos.

### Truncamiento de las colas de la distribución

El escenario principal excluye el **0,05 %** de la distribución conjunta ajustada de perturbaciones diarias antes de truncarla.

Los análisis de sensibilidad utilizan exclusiones del **0,2 %** y del **0,005 %**.

Estos porcentajes son hipótesis configurables de los escenarios, no probabilidades estimadas a partir de los datos ni límites aplicados al horizonte completo de previsión.

Para cada porcentaje, el radio se calcula a partir de los grados de libertad ajustados y del número efectivo de dimensiones de los factores aleatorios, incluidos los factores de divisas necesarios.

Los factores perfectamente dependientes o constantes no añaden dimensiones independientes.

Matemáticamente:

```text
radio = sqrt(
    d * (df - 2) / df
    * F_inverse_survival(objetivo, d, df)
)
```

Después del truncamiento, se recupera la covarianza histórica utilizando el segundo momento radial condicional exacto.

El informe registra:

- La proporción objetivo de probabilidad excluida y la proporción realmente excluida.
- Los grados de libertad ajustados.
- La dimensión efectiva.
- El radio calculado.
- El factor de reescalado de la varianza.
- El radio efectivo después del reescalado.

Cuando los datos son completamente deterministas, no es posible identificar los grados de libertad.

En ese caso, las perturbaciones permanecen en cero, `df` toma el valor `null` y la exclusión real de probabilidad es cero.

**No se asigna ningún valor ficticio a `df`.**

Los antiguos archivos de configuración de Monte Carlo con esquema de versión 1 deben migrarse a estos parámetros basados en probabilidades.

El programa los rechaza en lugar de reinterpretarlos silenciosamente.

Los informes anteriores que ya se hayan guardado no se modifican.

La configuración de las estadísticas descriptivas permanece sin cambios.

### Comparación de los resultados de los modelos

Que dos distribuciones tengan la misma media logarítmica y la misma covarianza no significa que produzcan el mismo patrimonio esperado.

### Moneda de referencia y conversión de divisas

La moneda utilizada para presentar los resultados procede de la configuración de la cartera guardada durante el procesamiento.

Las definiciones de tipos de cambio (`FX`) especifican:

- El identificador de la solicitud (`request ID`).
- La moneda base.
- La moneda cotizada.

Por ejemplo, en `EURUSD`, la cotización expresa cuántos dólares estadounidenses (`USD`) equivalen a un euro (`EUR`).

Las rentabilidades logarítmicas de los tipos de cambio necesarios se calibran y simulan conjuntamente con las rentabilidades de los activos, utilizando los mismos intervalos históricos válidos.

Para posiciones denominadas en dólares cuyos resultados se expresan en euros:

```text
EUR_log_growth = USD_log_growth - EURUSD_log_growth
```

Se admiten conversiones directas e inversas, pero no conversiones trianguladas mediante una tercera moneda.

Si el tipo de cambio necesario para convertir a la moneda de presentación no existe, no es válido o tiene un historial demasiado corto, el motor intenta expresamente utilizar `USD`.

En ese caso, una cartera compuesta exclusivamente por activos denominados en dólares puede simularse sin necesidad de conversiones adicionales.

Sin embargo, cualquier posición denominada en otra moneda sigue necesitando su conversión a dólares.

El motor nunca agrega importes expresados en monedas diferentes sin convertirlos previamente.

Las estadísticas históricas descriptivas mantienen por separado su criterio de cálculo en moneda local.

### Medidas de riesgo de cada escenario

Cada escenario guarda las siguientes medidas:

- **VaR (Value at Risk):** valor en riesgo por pérdidas durante el horizonte de previsión.
- **Expected Shortfall (ES):** pérdida esperada en los escenarios que superan el umbral del VaR.
- **Cuantiles del patrimonio final:** valores que delimitan distintos porcentajes de los resultados simulados.
- **Probabilidad de pérdida:** proporción de trayectorias que terminan con pérdidas.
- **Drawdown:** caídas del patrimonio desde máximos anteriores.
- **Media y mediana del patrimonio:** valores centrales de los resultados simulados.
- **Error estándar de Monte Carlo de la media final:** medida de la precisión de la estimación obtenida mediante simulación.

Los niveles de confianza del VaR y del Expected Shortfall son configurables y, por defecto, se utilizan el **95 % y el 99 %**.

Estas medidas corresponden al horizonte completo de previsión, no a un VaR diario.

Las pérdidas se expresan como fracciones del patrimonio inicial.

El VaR y el Expected Shortfall pueden resultar negativos cuando el umbral correspondiente representa una ganancia.

El error de muestreo de Monte Carlo es distinto de la incertidumbre sobre la tendencia histórica o sobre las hipótesis del modelo.

### Tratamiento de pérdidas numéricamente insignificantes

Las probabilidades de pérdida al final del horizonte y en horizontes cortos utilizan la misma regla para identificar valores numéricamente equivalentes a cero.

Las pérdidas relativas que no superan 64 veces el épsilon de máquina de `float64` —aproximadamente `1.42e-14` del patrimonio inicial— no se consideran pérdidas económicas.

Esto evita que pequeños residuos de precisión numérica, producidos por una posición en un activo y su divisa que se compensan exactamente, aparezcan como una probabilidad de pérdida del 100 %.

Para un patrimonio inicial de 100, la tolerancia equivale aproximadamente a `1.42e-12` unidades de patrimonio.

Las trayectorias, rentabilidades, valores de VaR/ES y momentos estadísticos no cambian.

Esta regla no establece una pérdida mínima de inversión ni modifica el modelo simulado.

### Precisión de los niveles de confianza

Los sufijos de las columnas correspondientes a cuantiles y niveles de confianza conservan el valor decimal completo de cada probabilidad configurada.

Por ejemplo, `0.95` y `0.9500001` generan columnas diferentes.

Los nombres habituales, como `loss_var_0.99`, se mantienen sin cambios.

Los informes y las etiquetas de confianza de los gráficos utilizan los mismos valores.

### Diagnósticos y comparaciones de estrés

Los diagnósticos y las comparaciones de estrés son opciones adicionales del esquema de configuración de versión 2.

Su configuración es:

```json
"diagnostics": {
  "enabled": true,
  "horizons_sessions": [1, 5, 20],
  "sigma_thresholds": [3, 4],
  "histogram_bins": 100
},
"stress_calibrations": [
  {
    "name": "gfc",
    "start": "2008-01-01",
    "end": "2009-06-30",
    "drift": "baseline"
  }
]
```

La calibración habitual de diez años se denomina `baseline`.

Cada periodo de estrés se configura mediante JSON.

Las fechas de las crisis y de los productos financieros no están fijadas directamente en el código Python.

Los dos extremos de cada intervalo de rentabilidad utilizado en un escenario de estrés deben estar comprendidos dentro del periodo configurado.

Se aplica el mismo requisito mínimo de observaciones históricas comunes.

Los periodos que terminan después de la fecha COB no están disponibles, de manera que no pueden introducirse observaciones futuras en la calibración de estrés.

Los informes muestran las fechas realmente utilizadas, el número de observaciones y las comparaciones que no hayan podido realizarse.

Las fechas de ejemplo son elecciones ilustrativas para los escenarios, no límites oficiales de las crisis financieras.

El módulo de estadísticas descriptivas dispone de sus propios periodos, configurados de forma independiente.

### Cómo se utiliza la tendencia en los escenarios de estrés

La opción:

```json
"drift": "baseline"
```

conserva la media logarítmica diaria de referencia de cada factor y su error estándar de estimación.

Sin embargo, sustituye la matriz de covarianzas por la estimada durante el periodo de estrés.

Esto modifica tanto las volatilidades como las correlaciones.

Los grados de libertad de la distribución t de Student se ajustan utilizando las rentabilidades del periodo de estrés centradas respecto a su propia media, no respecto a la media de referencia conservada.

La media muestral del periodo de estrés se exporta por separado de la tendencia utilizada en la simulación.

La opción:

```json
"drift": "stress"
```

utiliza, en cambio, la media muestral del periodo de estrés como tendencia simulada.

Mantener constante la tendencia logarítmica no implica mantener constante el patrimonio aritmético esperado.

En el modelo gaussiano, el crecimiento esperado también depende de la mitad de la varianza logarítmica.

### Comparación entre el escenario base y los escenarios de estrés

Cada comparación utiliza las mismas posiciones, el mismo horizonte de previsión y la misma moneda efectiva de simulación que el escenario de referencia.

Si faltan datos históricos o tipos de cambio necesarios para el periodo de estrés, esa comparación se declara no disponible.

El programa no puede cambiar silenciosamente la moneda ni eliminar posiciones de la cartera.

El escenario base que haya terminado correctamente se conserva.

Los modelos con el mismo nombre comparten secuencias de números aleatorios entre las distintas calibraciones.

Añadir comparaciones de estrés o activar los diagnósticos no modifica las trayectorias del escenario base.

Los parámetros de estrés permanecen constantes durante todo el horizonte de previsión.

Por tanto, se representa un entorno de riesgo sometido a estrés, no una reproducción cronológica de una crisis ni un modelo de agrupación temporal de la volatilidad.

Para desactivar las comparaciones de estrés, se utiliza un array vacío:

```json
"stress_calibrations": []
```

### Diagnósticos de las simulaciones

El módulo `src/mc_diagnostics.py` examina el patrimonio ya simulado sin generar nuevas perturbaciones aleatorias.

Exporta las siguientes estadísticas de las rentabilidades logarítmicas diarias agrupadas:

- Momentos estadísticos.
- Asimetría.
- Curtosis en exceso.
- Cuantiles.
- Frecuencia de superación de determinados umbrales.

Las puntuaciones estandarizadas (`z-scores`) de cada posición utilizan los momentos de la calibración.

Las puntuaciones estandarizadas de la cartera utilizan los momentos del conjunto de simulaciones, porque los pesos de una estrategia de comprar y mantener (*buy and hold*) cambian con el tiempo.

Las comparaciones históricas de cada posición utilizan la muestra real de calibración, expresada en la moneda de simulación, y sus propios momentos estadísticos.

La comparación histórica no presenta una cartera sintética de pesos constantes como si fuera un historial real de una estrategia de comprar y mantener.

Los histogramas cubren el intervalo de puntuaciones estandarizadas de `z=-10` a `z=10`.

Se conserva el recuento de las observaciones situadas fuera de ese intervalo y se utiliza la muestra completa como denominador de la densidad.

No se recortan los datos.

Las puntuaciones estandarizadas no están definidas cuando los datos son deterministas.

### Riesgo en horizontes cortos

El VaR y el Expected Shortfall de horizontes cortos utilizan las rentabilidades comprendidas entre el patrimonio inicial y la sesión bursátil configurada.

Se obtiene un resultado por cada trayectoria.

No se trata de rentabilidades móviles solapadas.

Si el horizonte solicitado supera la fecha final de previsión, se indica expresamente que no está disponible.

Los diagnósticos de riesgo de las trayectorias registran por separado:

- La peor rentabilidad de un solo día.
- La máxima caída acumulada desde un máximo anterior (*maximum drawdown*).

Los gráficos comparan, para cada calibración, las colas de las distribuciones diarias, las pérdidas a corto plazo y las distribuciones empíricas de las peores pérdidas diarias y de los máximos drawdowns.

Estos análisis complementan la distribución del patrimonio final a largo plazo, pero no modifican el modelo de simulación.

Las configuraciones anteriores del esquema de versión 2 pueden omitir estas dos opciones adicionales.

En ese caso, los diagnósticos se activan por defecto con los valores indicados anteriormente y no se realiza ninguna comparación de estrés.

### Archivos de resultados de Monte Carlo

Los resultados de la simulación se guardan por separado de los datos originales y de los datos ya limpiados.

La estructura de carpetas es:

```text
processed/<COB>/<processing-id>/monte_carlo/<simulation-id>/
```

En esta ruta:

- `COB` identifica la fecha de referencia de los datos.
- `processing-id` identifica la ejecución del procesamiento.
- `simulation-id` identifica una simulación concreta de Monte Carlo.

Cada simulación puede generar los siguientes archivos:

**1. Archivos de calibración**

- `calibration_returns.csv`: muestra histórica común exacta utilizada para calibrar los modelos.
- `calibration_parameters.csv`: medias logarítmicas diarias, volatilidades y errores estándar de las medias.
- `calibration_covariance.csv`: matriz de covarianzas de los factores simulados.

**2. Diagnósticos de la simulación**

- `simulation_diagnostics.csv`: compara las medias y covarianzas obtenidas mediante simulación con los valores objetivo.

**3. Resumen de riesgos**

- `risk_summary.csv`: contiene las medidas de riesgo de las posiciones y de la cartera, así como comprobaciones analíticas de las medias del modelo gaussiano.

**4. Resultados finales de las trayectorias**

- `terminal_values.csv`: contiene todos los resultados finales simulados, las máximas caídas acumuladas (*maximum drawdowns*) y las peores rentabilidades diarias.

**5. Diagnósticos diarios**

- `daily_distribution_diagnostics.csv`: estadísticas de las distribuciones históricas y simuladas, incluidos momentos y colas.
- `daily_histograms.csv`: histogramas estandarizados y recuentos de observaciones fuera de sus intervalos.

**6. Riesgo a corto plazo**

- `horizon_risk.csv`: VaR, Expected Shortfall y probabilidades de pérdida para los horizontes cortos configurados.
- `path_risk.csv`: distribuciones resumidas de la peor rentabilidad diaria y del máximo drawdown de cada trayectoria.

Todas las tablas identifican el conjunto de calibración mediante el campo `calibration_set`.

Los modelos de estrés utilizan un prefijo con el formato `<name>__`, donde `<name>` corresponde al nombre del escenario.

**7. Bandas y trayectorias**

- `bands.csv`: cuantiles y medias para cada fecha. Son bandas estadísticas marginales, no trayectorias individuales.
- `sample_paths.csv`: una pequeña selección configurable de trayectorias realmente simuladas.

**8. Informe, metadatos y gráficos**

- `report.md`: informe con las hipótesis y explicaciones de los resultados.
- `manifest.json`: archivo de metadatos de la ejecución.
- `plots/`: carpeta que contiene gráficos, como los gráficos de abanico (*fan charts*) y los histogramas de patrimonio final.

Los informes incluyen las decisiones relativas a la moneda utilizada, los detalles de los modelos y las sumas de comprobación (*checksums*) correspondientes.

### Reproducibilidad de las simulaciones

La semilla aleatoria (`seed`) permite reproducir los resultados.

Cada modelo utiliza su propia secuencia de números aleatorios, de modo que cambiar el orden de los modelos no modifica sus resultados.

La identidad de cada ejecución depende de:

- La configuración utilizada.
- Los datos de entrada.
- El código del programa.
- Las versiones de las dependencias.

Si se producen cambios numéricos relevantes, los resultados se guardan en una carpeta de salida diferente.

La simulación no utiliza conexiones a proveedores de datos mediante API, construcción de curvas OIS, calibración de opciones ni modelos neutrales al riesgo.

### Ejecutar Monte Carlo a partir de datos ya procesados

Es posible simular una ejecución previamente limpiada sin repetir las etapas anteriores.

Para ello, se puede utilizar el siguiente código Python:

```python
from pathlib import Path
from src import simulate_saved_processing

folder = Path("processed/2026-10-08/<processing-id>")

result = simulate_saved_processing(
    folder,
    "config/monte_carlo.json"
)

print(result.report())

result.save(
    folder / "monte_carlo" / result.run_id
)
```

La ruta del ejemplo debe sustituirse por la correspondiente a una ejecución real que ya exista.

El lector comprueba las sumas de verificación de los archivos de entrada y el número de filas.

También puede utilizarse `simulate_processed` con un objeto `ProcessedArchive` que ya esté cargado en memoria.

Si se reduce el número de trayectorias configuradas en `paths`, puede ser necesario reducir también `sample_paths`.

El número de trayectorias seleccionadas para mostrar nunca puede superar el número total de trayectorias simuladas.

## Limpieza y combinación de datos

El archivo `src/preprocessing.py` lee las respuestas originales guardadas y comprueba sus sumas de verificación (*checksums*), que permiten detectar si los datos han cambiado o están dañados.

El archivo `src/data_cleaning.py` procesa las observaciones por separado, sin modificar los archivos originales.

La configuración de este proceso se encuentra en `config/cleaning.json`, que incluye:

- Prioridades entre proveedores de datos.
- Calendarios de negociación de los mercados.
- Umbrales para detectar diferencias o errores.
- Configuraciones específicas opcionales para determinados instrumentos.

Se pueden incorporar nuevos símbolos bursátiles (*tickers*) sin modificar el código Python.

El programa aplica reglas diferentes según el tipo de información: acciones y ETF, índices, cotizaciones de opciones, datos históricos de opciones, operaciones corporativas, series de referencia y cotizaciones OIS introducidas manualmente.

### Orden del proceso de limpieza

**1. Lectura y verificación de las respuestas**

Se revisa cada intento de descarga registrado para cada proveedor y solicitud.

Los errores encontrados durante el inventario siguen siendo visibles. Las observaciones que no puedan leerse correctamente o que no superen la comprobación de integridad no pueden seleccionarse como datos válidos.

**2. Limpieza de los datos de cada proveedor**

Se comprueba que:

- Los números sean finitos y válidos.
- Las fechas estén dentro del período solicitado.
- Los datos tengan la frecuencia diaria esperada.
- Los precios sean positivos.
- Los valores de apertura, máximo, mínimo y cierre (`OHLC`) sean coherentes.
- No existan duplicados problemáticos ni inconsistencias con el calendario de mercado.

Los duplicados idénticos se consolidan. Si existen versiones contradictorias, se conservan, pero se ponen en cuarentena y no se utilizan automáticamente.

También se aíslan las discrepancias explícitas entre las monedas o unidades configuradas y las declaradas por el proveedor.

Se conservan ambos valores originales, sin convertir automáticamente monedas ni unidades.

**3. Selección del proveedor y del tipo de precio**

Para cada instrumento se selecciona un proveedor principal y una base de precios.

En acciones y ETF se prefieren los precios de cierre ajustados por desdoblamientos (*splits*) y dividendos, siempre que el proveedor documente cómo realiza esos ajustes.

Los índices conservan su convención declarada, ya sea de precios o de rentabilidad total.

Cada precio ajustado seleccionado debe contar con una convención de ajuste documentada.

También se conservan los precios de cierre ordinarios y las operaciones corporativas.

**4. Comparación entre proveedores**

Los precios de cierre ordinarios solo se comparan cuando coinciden la moneda, las unidades y la base de ajuste.

El programa nunca calcula una media de los precios procedentes de distintos proveedores.

El umbral inicial de discrepancia relativa es del **0,1 %**.

Los huecos en una serie de precios ordinarios pueden completarse con datos de otro proveedor únicamente cuando:

- Los precios utilizan la misma base.
- Existen al menos dos observaciones coincidentes entre proveedores.
- Esas observaciones concuerdan dentro de la tolerancia permitida.
- Ninguna observación comparable incumple el umbral.

**Las series de precios de cierre ajustados nunca se empalman automáticamente.**

**5. Alineación de fechas y cálculo de rentabilidades**

Los datos se alinean con las sesiones reales de negociación.

Se calculan rentabilidades simples y logarítmicas únicamente entre sesiones consecutivas.

Cuando falta una observación, se mantiene el hueco: no se inventa un precio mediante interpolación ni se copia el precio anterior.

Si falta una sesión, la rentabilidad correspondiente al siguiente intervalo de una sesión deja de ser válida.

Para construir la tabla común de rentabilidades de la cartera, todas las posiciones deben compartir las mismas fechas de inicio y final de cada intervalo.

Los intervalos incompatibles por diferencias entre calendarios se excluyen y se documentan.

Las fechas individuales de inicio de las rentabilidades se conservan en la tabla de precios seleccionados.

**6. Revisión independiente de las cotizaciones de opciones**

Las cotizaciones de opciones se examinan por separado.

Se comprueban, entre otros elementos:

- Precios de compra y venta (`bid` y `ask`) positivos y correctamente ordenados.
- Características del contrato y fecha de vencimiento.
- Volatilidad implícita del modelo (`IV`).
- Valor del activo subyacente utilizado por el modelo.
- Valor actual de los dividendos (`dividend PV`) no negativo, según la política configurada.

El límite inicial del diferencial relativo entre compra y venta es del **10 % del precio medio**.

La ausencia de determinadas sensibilidades (*Greeks*) o del volumen no implica automáticamente rechazar una cotización.

Si se proporciona una `vega` del modelo muy baja o negativa, se señala como posible problema.

Siempre se selecciona una observación completa: no se mezclan campos procedentes de diferentes proveedores o momentos.

**7. Conservación de información y generación de informes**

Se mantienen las fechas originales de las series de referencia, las unidades de los tipos de interés, las operaciones corporativas y las fechas de observación de las cotizaciones OIS.

Finalmente, se guardan las tablas procesadas, los datos excluidos, las comparaciones entre proveedores y un informe del proceso.

### Calendarios de mercado

El calendario predeterminado es `XNYS`, correspondiente a la Bolsa de Nueva York, mediante la biblioteca `exchange-calendars`.

Este calendario contempla festivos históricos, cierres extraordinarios y sesiones con cierre anticipado.

Para instrumentos negociados en otros mercados debe configurarse el calendario correspondiente.

El calendario bursátil de acciones no debe aplicarse automáticamente a los tipos de interés, las divisas o las horas de captura de cotizaciones de opciones.

Las fechas anteriores a la primera observación disponible de un instrumento se muestran como períodos sin cobertura, no como precios inventados.

### Tratamiento de dividendos y precios ajustados

Las rentabilidades calculadas a partir de precios de cierre ajustados ya incorporan los ajustes por distribuciones realizados por el proveedor.

**No deben sumarse otra vez los dividendos guardados a esas rentabilidades**, porque se contabilizarían dos veces.

Los dividendos históricos tampoco constituyen una previsión de dividendos futuros.

El valor actual de dividendos utilizado en el modelo de opciones no equivale a un calendario previsto de pagos futuros.

### Consideraciones sobre los datos de opciones

La volatilidad implícita del modelo, el precio del modelo, el valor del subyacente y el valor actual de dividendos se conservan por separado de los precios `bid`, `ask`, el precio medio y las volatilidades implícitas obtenidas a partir de esos precios.

La volatilidad implícita se expresa como un número decimal anualizado.

La hora de vencimiento configurada como `AM` o `PM` se identifica expresamente como una estimación basada en una convención.

La marca temporal indica cuándo el programa recibió la información del proveedor, pero no demuestra que la cotización estuviera actualizada en ese momento en el mercado.

Si la captura de las cotizaciones se extiende durante más de cinco minutos, se genera una advertencia.

Que una cotización sea candidata para el análisis no garantiza que:

- Sea simultánea a otras cotizaciones.
- Tenga suficiente liquidez.
- Sea coherente con los precios a plazo.
- Esté libre de oportunidades de arbitraje.

Las opciones americanas se distinguen de las candidatas utilizadas para construir superficies de volatilidad de opciones europeas.

No se inventan volatilidades implícitas ni sensibilidades (*Greeks*) para las barras históricas de opciones.

### Archivos de salida

Los datos originales descargados se conservan en:

`tests/data_dumps/YYYY-MM-DD/`

Los informes del inventario se guardan en:

- `reports/YYYY-MM-DD/inventory.json`
- `reports/YYYY-MM-DD/inventory.md`

Los resultados derivados del procesamiento se guardan en:

`processed/YYYY-MM-DD/<processing-id>/`

En estas rutas, `YYYY-MM-DD` representa la fecha en formato año-mes-día, mientras que `<processing-id>` identifica una ejecución concreta del procesamiento.

| Archivo | Contenido |
|---|---|
| `provider_prices.csv` | Observaciones de precios de cada proveedor, indicadores de calidad y procedencia de los duplicados. |
| `selected_prices.csv` | Precios seleccionados y alineados con el calendario, proveedor de origen, base de precios, rentabilidades simples y logarítmicas y períodos sin datos. |
| `aligned_holding_returns.csv` | Intervalos de fechas coincidentes con rentabilidades válidas en moneda local para todas las posiciones de la cartera. |
| `price_comparisons.csv` | Diferencias entre precios comparables de distintos proveedores e indicadores de cumplimiento de las tolerancias. |
| `corporate_actions.csv` | Distribuciones y desdoblamientos de acciones (*splits*), conservados por separado de las rentabilidades ajustadas. |
| `option_quotes.csv` | Todas las observaciones de opciones tras la limpieza, exclusiones, volatilidades implícitas, sensibilidades (*Greeks*) y campos del modelo. |
| `selected_option_quotes.csv` | Observaciones completas seleccionadas, incluidas las candidatas rechazadas para poder examinarlas. |
| `option_iv_candidates.csv` | Cotizaciones preliminares que superan las comprobaciones de campos, diferencial de precios y `vega`. |
| `historical_option_bars.csv` | Datos históricos de precios de opciones, separados de las cotizaciones puntuales (*snapshots*). |
| `reference_observations.csv` y `selected_references.csv` | Observaciones originales de tipos de interés y divisas, y selección de sus proveedores. |
| `ois_quotes.csv` | Cotizaciones OIS a la par, expresadas en porcentaje y en formato decimal, con sus fechas reales. |
| `exchange_sessions.csv` | Fechas y horas de apertura y cierre de las sesiones de negociación. |
| `request_status.csv` e `issues.csv` | Estados de las solicitudes guardadas y diagnósticos de errores de lectura o procesamiento. |
| `manifest.json` y `report.md` | Política de procesamiento, cartera actual y archivada, cobertura de datos, preparación para el análisis y huellas de integridad de los archivos generados. |

### Identificación y conservación de los resultados procesados

Cada identificador de procesamiento (`processing-id`) se calcula a partir de los datos de entrada, sus sumas de verificación, las reglas de limpieza, la cartera y la versión del calendario bursátil.

Si cambian los pesos de la cartera o las reglas de limpieza, se genera una ejecución derivada diferente.

Si los datos de entrada y la configuración son idénticos, se vuelve a utilizar la misma ubicación de salida.

**Las respuestas originales de los proveedores nunca se modifican.**

Que el procesamiento haya terminado correctamente significa que se ha completado el procedimiento, no que todos los datos hayan superado las comprobaciones de calidad.

### Uso de la configuración actual de la cartera

El procesamiento utiliza el archivo JSON de la cartera **actual**, pero conserva en el informe la configuración de la cartera archivada originalmente.

Para aplicar nuevos pesos a un archivo histórico ya descargado, se debe elegir la **opción 1** del programa.

Las reglas para reutilizar archivos descargados siguen siendo estrictas: si cambia la cartera o la configuración, el modo de descarga puede rechazar la reutilización de un archivo anterior.

El procesamiento local no evita esas comprobaciones ni realiza nuevas solicitudes a los proveedores.

### Operaciones que no realiza la limpieza

La etapa de limpieza no realiza:

- Valoración monetaria de la cartera.
- Conversión de divisas.
- Rebalanceo de posiciones.
- Previsión de dividendos.
- Calibración de SPY a partir de SPX.
- Construcción de curvas OIS (*bootstrap*).
- Ajuste de superficies de volatilidad.
- Simulaciones de Monte Carlo.

Para analizar en euros posiciones denominadas en monedas extranjeras, se necesita disponer del historial correspondiente de tipos de cambio.

### Utilización desde una biblioteca Python

Es posible ejecutar directamente el procesamiento de un archivo descargado mediante:

```python
from src import process_archive

result = process_archive(
    "tests/data_dumps/2026-10-08",
    "config/cleaning.json",
    portfolio_file="config/portfolio.json",
    progress=print,
)

result.save(f"processed/{result.cob}/{result.run_id}")
```

El objeto `result.tables` contiene tablas de datos de `pandas` (*DataFrames*).

El objeto `result.summary` contiene información de diagnóstico sobre el procesamiento.

Para trabajar con carpetas situadas fuera del directorio del proyecto, deben utilizarse rutas absolutas.

También puede ejecutarse el punto de entrada del programa de esta manera:

```python
run.main("08/10/2026", mode="process")
```

El alias antiguo `mode="inventory"` también realiza ahora la limpieza y combinación de datos.

Por compatibilidad, devuelve el objeto de inventario.

El modo de descarga devuelve el resultado original de la descarga.

Cuando se necesiten directamente las tablas procesadas, se recomienda utilizar `process_archive()`.

## Modo de descarga: una ejecución, un archivo por fecha COB

Para descargar datos, selecciona la **opción 2 - Download** al ejecutar `run.py` con el intérprete del entorno `.venv` del proyecto, desde Visual Studio Code o Python.

El programa muestra la fecha actual según la zona horaria de **Europa/Londres** y solicita dos fechas:

- **COB:** fecha de cierre de referencia de los datos (*Close of Business*).
- **Fecha de inicio del historial:** primer día desde el que se desea solicitar información histórica.

Ambas fechas se introducen en formato `DD/MM/YYYY`.

La fecha COB debe ser, como máximo, la del día anterior. La fecha de inicio del historial debe ser anterior o igual a COB.

### Funcionamiento de la descarga

El programa sigue estos pasos:

1. Si ya existe un archivo completo y compatible en `tests/data_dumps/YYYY-MM-DD`, lo reutiliza sin realizar nuevas solicitudes a las API, aunque el programa se ejecute otro día.
2. Si no existe un archivo compatible, solicita los historiales de SPX, SPY, IEF y las series de referencia habilitadas de FRED y ECB, desde la fecha inicial seleccionada hasta COB.
3. Para las opciones financieras se aplica una regla especial según la fecha COB. Todas las respuestas y los errores se guardan en el archivo correspondiente.
4. El archivo nuevo o reutilizado se inventaría, limpia y combina automáticamente para generar resultados derivados separados. Después termina la ejecución.

En este flujo no se realiza calibración de superficies de volatilidad.

### Regla especial para las opciones de IBKR

El programa distingue entre una fecha COB correspondiente a ayer y otra más antigua.

| Fecha COB solicitada | Datos de opciones de IBKR | Fecha de observación |
|---|---|---|
| Ayer | Cotizaciones actuales (*live snapshots*), incluidas volatilidades implícitas y sensibilidades (*Greeks*) cuando IBKR las proporciona. | Fecha y hora reales de captura de hoy. |
| Anterior a ayer | Barras históricas de un minuto de `TRADES`, `BID`, `ASK` y `MIDPOINT`. | Únicamente la fecha COB solicitada. |

**Ejemplo del README original:**

Si hoy fuera el 8 de octubre de 2026:

- COB `07/10/2026`: se utilizarían cotizaciones actuales de opciones capturadas el 8 de octubre.
- COB `06/10/2026`: se solicitarían datos históricos de opciones correspondientes al 6 de octubre.

La regla utiliza días naturales.

Los fines de semana y festivos no se trasladan silenciosamente a otra fecha de negociación.

Si falla una solicitud histórica, el programa nunca la sustituye automáticamente por cotizaciones actuales para una fecha COB antigua.

### Advertencia sobre las fechas de los datos

**Un archivo cuya fecha COB sea ayer contiene deliberadamente precios históricos de activos hasta COB y cotizaciones actuales de opciones obtenidas al día siguiente.**

Por tanto, esos datos no constituyen una superficie histórica sincronizada de opciones.

Cada cotización actual registra por separado su fecha real de captura.

El adaptador comprueba la fecha de Londres antes de cada lote de solicitudes y deja de iniciar nuevas capturas si cambia el día.

## Datos de entrada y cartera

La cartera está formada por un **50 % de SPY y un 50 % de IEF**, con presentación de resultados en euros (`EUR`).

Esta configuración se encuentra en:

`config/portfolio.json`

El índice `SPX` y sus opciones se utilizan como datos de referencia para modelizar los movimientos de `SPY`.

**SPX y sus opciones no son posiciones de la cartera.**

La configuración incluye una relación entre opciones e instrumentos de referencia para reflejar esta finalidad. La calibración de esa relación queda pendiente para una etapa posterior.

### Proveedores y archivos de configuración

Los datos se solicitan mediante distintos proveedores, cada uno con su propio archivo JSON.

- **`yahoo.json`, `massive.json` e `ibkr.json`:** contienen solicitudes de datos históricos de SPX, SPY e IEF.
- **`alpha_vantage.json` y `marketdata.json`:** tienen habilitados SPX, SPY e IEF. Sus adaptadores actuales no pueden obtener el historial del índice SPX. Por ello, esas solicitudes registran `unsupported_operation` en JSON, sin intentar descargarlas como si fueran acciones. SPY e IEF utilizan solicitudes de historial de ETF. SPY aporta datos adicionales, pero no sustituye a SPX.
- **`fred.json` y `ecb.json`:** contienen las series de referencia de tipos de interés, tipos de interés a un día (*overnight rates*) y tipos de cambio.
- **`ibkr.json`:** incluye los **4.181 contratos explícitos de opciones de compra (*call*) SPX AM**, con 14 vencimientos hasta el **17 de diciembre de 2027**. No se incluyen opciones semanales `SPXW`, opciones de venta (*put*) ni búsquedas automáticas de nuevos contratos durante la ejecución.
- Los demás archivos JSON de proveedores no contienen solicitudes de opciones en la ejecución normal.

### Identificadores de los instrumentos

Cada proveedor utiliza sus propios identificadores.

Por ejemplo:

- Yahoo utiliza `^GSPC` para el índice S&P 500.
- Massive utiliza `I:SPX`.
- IBKR utiliza contratos estructurados que especifican el símbolo, el tipo de instrumento y el mercado.

Cada archivo JSON contiene las instrucciones y los detalles de entrada correspondientes.

La búsqueda de contratos de IBKR valida las opciones que ya han sido seleccionadas en la configuración; no descubre automáticamente otras opciones.

### Selección de fechas para descargar datos

Después de seleccionar la opción de descarga, `run.py` solicita:

- La fecha **COB**.
- La fecha de **inicio del historial**.

Ambas deben introducirse en formato `DD/MM/YYYY`.

La fecha inicial debe ser anterior o igual a COB y no puede ser anterior al **1 de enero de 1900**.

Las solicitudes JSON de instrumentos distintos de opciones utilizan:

```json
{"start": "HISTORY_START", "end": "COB"}
```

`HISTORY_START` y `COB` son marcadores que el programa sustituye en memoria por las fechas introducidas.

Las fechas resueltas se guardan en:

`resolved_inputs.json`

Los archivos JSON originales de configuración **no se reescriben cada vez que se ejecuta el programa**.

También puede utilizarse la biblioteca desde Python:

```python
MarketDataDownloader.from_json(
    ...,
    cob_date=date(...),
    history_start=date(...)
)
```

Si un archivo JSON personalizado contiene fechas explícitas, estas se respetan cuando no se proporciona una fecha inicial que las sustituya durante la ejecución.

### Solicitudes históricas y recuperación de datos disponibles

La configuración antigua `lookback: "max"` se rechaza si no se proporciona una fecha de inicio.

En ese caso, el programa muestra un mensaje indicando que debe actualizarse la configuración.

No selecciona automáticamente enero del año 2000.

Primero se solicita el intervalo de fechas elegido por el usuario.

Si los datos disponibles comienzan más tarde, se conserva únicamente la parte accesible, sin rellenar las fechas ausentes.

Si el proveedor devuelve un error relacionado con el intervalo de fechas o con las restricciones del plan contratado, el programa puede intentar recuperar el máximo historial accesible mediante un procedimiento específico para ese proveedor.

Cuando la API admite límites de fechas, la recuperación sigue estando limitada por COB.

Los siguientes errores **no provocan automáticamente una solicitud con un intervalo más amplio**:

- Ausencia de credenciales.
- Errores de autenticación.
- Fallos de conexión o transporte.
- Restricciones de frecuencia de solicitudes (*pacing*).

Los proveedores que no admiten determinados índices siguen registrando ese problema expresamente.

### Procedimiento de recuperación por proveedor

**Yahoo**

Busca la fecha de la primera negociación disponible y solicita los datos hasta COB.

**FRED y ECB**

Eliminan el límite inferior de fechas y mantienen COB como límite superior.

**Alpha Vantage**

Solicita el historial diario completo.

Si se deniega esa modalidad, intenta obtener el historial reducido (*compact*).

Este servicio no admite fechas de inicio y fin en esa operación. Por tanto, los datos originales pueden incluir observaciones fuera del intervalo solicitado o no cubrir una fecha COB antigua.

**MarketData.app**

Recupera páginas de datos hacia atrás en el tiempo mediante `countback`, hasta COB.

Reduce el tamaño de las páginas cuando encuentra límites de acceso.

**Massive**

Solicita datos hacia atrás mediante intervalos de fechas y reduce los intervalos rechazados.

La búsqueda de datos agregados se detiene en el límite de marcas temporales Unix no negativas (1970) o en el primer intervalo que no esté disponible.

**IBKR**

Utiliza fragmentos diarios dentro del intervalo seleccionado, respetando una posible primera fecha disponible posterior.

Si falla una solicitud por su intervalo de fechas, vuelve a intentarlo desde la primera fecha comunicada por el proveedor.

Si esa consulta también falla, busca hacia atrás utilizando el año 1900 como límite de seguridad.

No existe una sustitución automática por enero del año 2000.

### Limitaciones del historial recuperado

Cuando el programa habla del historial «máximo», se refiere al **máximo historial accesible a través del proveedor**, no a una garantía de cobertura completa.

Se registran:

- Los primeros intervalos que no están disponibles.
- Las restricciones de la cuenta o del plan.
- Los límites de seguridad de las búsquedas.

Los huecos y la cobertura de fechas anteriores pueden permanecer sin verificar.

Se conservan la solicitud original, el motivo de la recuperación alternativa y los intervalos realmente consultados, junto con los metadatos de estado y resumen.

Las respuestas originales permanecen sin modificar.

Una recuperación alternativa puede incluir observaciones anteriores a la fecha inicial solicitada.

En esta etapa no se limpian, recortan ni rellenan los datos.

Los archivos históricos compatibles que contienen intervalos más amplios pueden reutilizarse sin descargarlos de nuevo ni modificar sus archivos originales.

Sin embargo, si se solicita un historial anterior al que identifica el archivo guardado, se rechaza su reutilización en lugar de ampliarlo silenciosamente.

### Solicitudes de opciones de IBKR

Las opciones de IBKR conservan la operación:

`operation: "option_history"`

Y las fechas de entrada:

```json
{"start": "COB", "end": "COB"}
```

El archivo `src/cob.py` selecciona el procedimiento de consulta histórico o actual según la fecha.

Los parámetros históricos especifican:

- Barras de un minuto.
- Canales de datos solicitados.
- Horario habitual de negociación.
- Hora límite de las 16:00 en la zona `America/New_York`.

La fecha de vencimiento de una opción es independiente de la fecha COB.

Solo se solicitan los contratos incluidos expresamente en la configuración, también cuando se consultan fechas antiguas.

La biblioteca no puede deducir automáticamente qué contratos estaban realmente admitidos a negociación en una fecha histórica.

## Datos OIS introducidos manualmente

El archivo `config/ois.json` contiene cotizaciones OIS introducidas manualmente, correspondientes a:

- **USD SOFR:** tipos de referencia vinculados al dólar estadounidense.
- **EUR ESTR:** tipos de referencia vinculados al euro.

Estas cotizaciones proceden del documento `OIS.docx` y tienen fecha del **9 de octubre de 2026**.

Se incluyen:

- **86 plazos SOFR**, desde una semana (`1W`) hasta 50 años (`50Y`).
- **44 plazos ESTR**, desde una semana (`1W`) hasta 60 años (`60Y`).

Se trata de cotizaciones de swaps OIS a la par (*par OIS swap quotes*) con inicio inmediato (*spot-starting*).

Únicamente se utiliza la columna `Live` del documento original para construir las curvas.

Las columnas de comparación histórica quedan excluidas.

### Unidades de los tipos de interés

Los tipos de interés se expresan como porcentajes anuales.

Por ejemplo:

```text
3.890 significa 3,890 % anual.
```

Estas cotizaciones no son tipos cero cupón (*zero rates*).

Cuando el documento original no proporciona determinadas convenciones de los contratos, sus valores se mantienen como `null`.

La construcción de curvas mediante *bootstrap*, los factores de descuento y los tipos de interés a plazo (*forwards*) quedan pendientes para una fase posterior.

### Incorporación de los datos OIS al programa

El archivo `run.py` transmite la variable `OIS_FILE` a la función:

```python
download_cob(..., ois_file=...)
```

Antes de realizar solicitudes a los proveedores, la función:

```python
src.ois.load_ois_config()
```

valida los datos de entrada.

Para incorporar estos datos OIS no se necesita:

- Una suscripción.
- Una clave de API.
- Una sesión de navegador.
- Una conexión a una API.

El archivo JSON de OIS se mantiene separado de los archivos de solicitudes a proveedores.

Por ello, la carga automática de proveedores desde una carpeta excluye `ois.json`.

### Política de fechas de las curvas OIS

Cada región contiene exactamente una curva fija.

Con la configuración:

```text
date_policy=fixed_for_all_cobs
```

todas las fechas COB utilizan las mismas cotizaciones SOFR y ESTR, que conservan su fecha de observación original del **09/10/2026**.

El archivo de datos registra:

```text
usage_policy=fixed_for_all_cobs
```

También indica si la fecha de observación de las cotizaciones es anterior, posterior o igual a la fecha COB solicitada.

El indicador:

```text
suitable_for_strict_cob
```

solo es verdadero cuando ambas fechas coinciden.

Este indicador sirve para documentar la procedencia y coherencia temporal de los datos; no impide incluir las curvas fijas cuando las fechas son diferentes.

No se realiza una selección de curvas por fecha, ni una actualización automática, ni se excluyen las curvas porque falten cotizaciones correspondientes a otra fecha.

### Archivos OIS generados

Los nuevos archivos de datos incluyen:

- `ois/input.json`: configuración de entrada utilizada.
- `ois/us/curve.json`: curva OIS de Estados Unidos.
- `ois/us/status.json`: estado de los datos estadounidenses.
- `ois/euro/curve.json`: curva OIS de la zona euro.
- `ois/euro/status.json`: estado de los datos europeos.

Los resultados de ambas curvas también aparecen en el resumen principal, incluso cuando falla alguna API de otro proveedor.

### Comprobaciones para reutilizar archivos

Los datos OIS forman parte de las comprobaciones de identidad de cada archivo histórico.

Si se modifican las cotizaciones OIS o se añaden datos OIS a un archivo anterior, el programa rechaza su reutilización en lugar de sobrescribirlo.

Los archivos completos y compatibles se comprueban y leen sin nuevas solicitudes a las API ni escrituras adicionales.

### Conservación de las curvas originales

En `config/ois.json` debe mantenerse exactamente una curva fija por región, conservando:

- La fecha original de observación.
- Las unidades expresadas en porcentaje.
- El tipo de cotización.

El archivo JSON contiene todos los datos OIS necesarios para ejecutar el programa.

Por tanto, **no es necesario disponer de `OIS.docx` para ejecutar el programa ni para realizar pruebas sin conexión a internet**.

El nombre y la suma de verificación del documento original se conservan únicamente para identificar la fuente.

La comparación con ese documento se realizó cuando se importaron los tipos de interés.

Las pruebas actuales comprueban la validación del JSON, la inclusión de las curvas fijas y la reutilización de los archivos.

Los metadatos relacionados con los gráficos de tipos a plazo se conservan únicamente para documentar su procedencia.

No se extrajeron cotizaciones numéricas de tipos a plazo.

## Limitaciones de los datos históricos y actuales de opciones

### Solicitudes de datos históricos

Las solicitudes históricas de IBKR utilizan la función `reqHistoricalData`, con las siguientes características:

- Una hora de finalización explícita en formato UTC.
- Una duración de un día.
- El parámetro `keepUpToDate=False`.

Las barras históricas de opciones de IBKR **no proporcionan volatilidades implícitas ni sensibilidades (*Greeks*) históricas para cada contrato**.

El adaptador registra estos campos como no disponibles, en lugar de inventar sus valores.

IBKR no proporciona historiales de opciones que ya han vencido. Por tanto, que un contrato figure en la configuración no garantiza que sus datos históricos estén disponibles.

Las barras de precios intradía no equivalen a precios oficiales de liquidación.

Los canales históricos que no estén disponibles siguen apareciendo identificados en los resultados.

### Solicitudes de cotizaciones actuales

Las solicitudes actuales utilizan el procedimiento de `ib_insync` descrito en el tutorial del proyecto:

1. Se establece una conexión de solo lectura para opciones.
2. Se validan los contratos especificados expresamente.
3. Se solicitan cotizaciones puntuales mediante `reqTickersAsync`.

Se permiten como máximo **50 solicitudes simultáneas de opciones**.

Estas consultas obtienen datos actuales de mercado.

Las cotizaciones regulatorias de pago están desactivadas.

Cuando IBKR proporciona los datos, se conservan:

- Precios de compra y venta.
- Volatilidad implícita del modelo.
- Sensibilidades del modelo (*Greeks*).
- Valor actual de los dividendos.
- Precio del activo subyacente.

Cuando faltan campos del modelo, se registra el estado `partial_model_data`.

Tener una suscripción a datos de mercado no garantiza que todos los campos estén disponibles para todos los precios de ejercicio (*strikes*).

**El programa no calcula localmente volatilidades implícitas ni sensibilidades.**

### Límites de conexión y frecuencia de solicitudes

Las solicitudes históricas utilizan una conexión independiente de solo lectura.

Se procesan como máximo **cinco contratos simultáneamente**, respetando límites de frecuencia de solicitudes.

Una opción histórica puede necesitar cuatro solicitudes diferentes.

Si se producen tres errores consecutivos de tiempo de espera, límite de solicitudes o red en un canal, el programa detiene las solicitudes de ese canal durante el resto de la ejecución.

Estos fallos quedan registrados expresamente.

Las descargas de conjuntos muy grandes de contratos de opciones pueden requerir bastante tiempo.

### Configuración necesaria de IBKR

La aplicación **TWS o IB Gateway debe permanecer abierta y con la sesión iniciada**.

En TWS deben habilitarse:

- `ActiveX and Socket Clients`.
- `Read-Only API`, para permitir únicamente operaciones de lectura.

El puerto debe coincidir con `IBKR_PORT`, configurado actualmente como `7496`.

También debe utilizarse un identificador de cliente exclusivo, actualmente `81`.

La autenticación se realiza mediante TWS.

No existe una clave API de IBKR guardada en el archivo JSON.

Las credenciales de los proveedores HTTP se obtienen de las variables de entorno indicadas en `run.py`.

Si algún proveedor no está disponible, se registra la incidencia sin detener a los demás proveedores independientes.

**El proyecto no implementa solicitudes de envío de órdenes de compraventa.**

## Consulta y reutilización de los datos descargados

Cada carpeta identificada por una fecha contiene los siguientes archivos:

- **`manifest.json`:** fecha COB, fecha de ejecución, procedimiento utilizado para las opciones, identificador de la configuración y recuento de estados.
- **`summary.json`:** un resultado por cada proveedor y solicitud, incluidos el método de recuperación y los campos que faltan.
- **`resolved_inputs.json` y `portfolio.json`:** copia de los datos de entrada y de la configuración de la cartera utilizados.
- **Carpetas de proveedores y solicitudes:** contienen `request.json`, `status.json` y archivos `attempt_001/response_*`, junto con los metadatos de las respuestas.

### Conservación de los datos originales

Los cuerpos originales de las respuestas HTTP y las respuestas de IBKR serializadas se conservan antes de realizar la limpieza.

Si una respuesta incluye por error alguna credencial configurada, esta se oculta.

Los datos originales pueden conservar información adicional proporcionada por el proveedor, metadatos actuales y ajustes aplicados en origen.

Alpha Vantage no dispone de un parámetro para establecer la fecha final del historial en esa operación.

Por tanto, sus observaciones pueden extenderse más allá de COB y deben limitarse al intervalo correspondiente antes de realizar el análisis.

Las observaciones macroeconómicas corresponden a las versiones disponibles actualmente, no a un archivo histórico de revisiones de los datos.

### Significado del estado de una descarga

El estado `complete` significa que se han intentado todas las solicitudes configuradas.

**No significa que todas las solicitudes hayan proporcionado todos los datos esperados.**

Para comprobar los resultados deben revisarse:

- `all_requests_succeeded`: indica si todas las solicitudes finalizaron correctamente.
- Los estados de cada canal de datos.
- Los campos que faltan en las respuestas.

De forma predeterminada, si existe un archivo completo, el programa lo reutiliza, aunque contenga solicitudes fallidas.

Si una ejecución se interrumpe, el programa puede reanudar las operaciones que quedaron pendientes.

Si se intenta utilizar una configuración diferente para una fecha COB que ya tiene un archivo guardado, el programa rechaza la operación antes de realizar solicitudes a las API.

Esto protege los datos originales frente a modificaciones accidentales.

### Protección frente a ejecuciones simultáneas

El archivo de bloqueo `.running` impide que dos ejecuciones escriban simultáneamente en el mismo archivo de datos.

Si el programa se cierra de forma forzada, es necesario comprobar primero que el proceso anterior ya no sigue ejecutándose antes de eliminar un bloqueo antiguo.

### Compatibilidad entre archivos descargados

La versión 3 del esquema de archivos identifica tanto los parámetros efectivos de obtención de datos como las solicitudes originales.

Las credenciales y los parámetros de transporte quedan excluidos de esta identificación.

Por ello, cambiar una contraseña o una credencial no invalida automáticamente un archivo histórico.

Los archivos compatibles con el esquema 2 también pueden leerse y reutilizarse tal como fueron capturados originalmente.

Cada nuevo intento de descarga guarda la operación realmente ejecutada y las fechas solicitadas en su propio archivo:

`attempt_NNN/request.json`

Los metadatos de la solicitud y el resumen muestran el intervalo de fechas del último intento.

El archivo original `resolved_inputs.json` continúa identificando el conjunto de datos, de modo que un nuevo intento no borra el historial de las solicitudes anteriores.

### Comprobación de los intervalos de fechas

Al reutilizar datos, el programa también comprueba el intervalo de fechas realmente solicitado en cada intento guardado.

Por ejemplo, si una solicitud original era amplia, pero solo se consiguió descargar correctamente un intervalo más reducido, ese resultado no puede utilizarse automáticamente para satisfacer una nueva solicitud más amplia.

En ese caso, la reutilización predeterminada se rechaza sin realizar nuevas solicitudes ni modificar archivos.

Si se utiliza:

`retry_failed=True`

el programa puede recuperar el intervalo insuficiente mediante un nuevo intento, conservando los archivos anteriores.

Para reutilizar una descarga deben coincidir los instrumentos, los parámetros y las fechas correspondientes.

Un historial guardado más amplio sí puede aceptarse para satisfacer una solicitud más reducida.

El esquema 2 no registraba todos los parámetros efectivos de ejecución. Por ello, esos parámetros no pueden comprobarse retrospectivamente y el programa muestra un aviso en la consola.

Los archivos completos no se reescriben ni se descargan otra vez.

Los cambios de instrumentos o de los parámetros de ajuste registrados siguen provocando un rechazo por incompatibilidad.

La incorporación de nuevos símbolos o instrumentos (*tickers*) todavía no se realiza de forma incremental.

Los errores guardados siguen siendo visibles y solo se vuelven a intentar cuando se indica expresamente `retry_failed=True`.

### Reintento de solicitudes fallidas

Para volver a intentar las solicitudes fallidas desde Python:

```python
import run

result = run.main(
    "07/10/2026",
    "03/01/2000",
    mode="download",
    retry_failed=True
)
```

Los nuevos intentos conservan los archivos de las solicitudes anteriores.

También aplican la regla de fechas vigente en el momento del reintento.

Por ejemplo, una cotización actual de opciones que no pudo obtenerse ayer no puede recrearse como dato actual una vez transcurrido su día de captura.

Si se vuelve a intentar más tarde, el programa utilizará barras históricas y registrará expresamente su procedencia.

### Utilización directa desde Python

También es posible utilizar directamente la biblioteca:

```python
from datetime import date
from src import MarketDataDownloader

dl = MarketDataDownloader.from_json(
    "config",
    cob_date=date(2026, 10, 7),
    history_start=date(2000, 1, 3)
)

result = dl.download_cob(
    "07/10/2026",
    "tests/data_dumps",
    portfolio_file="config/portfolio.json",
    ois_file="config/ois.json"
)
```

Si se ejecuta desde un directorio de trabajo diferente, deben utilizarse rutas absolutas.

La biblioteca no llama nunca a `input()` para solicitar datos al usuario.

Solo `run.py` muestra preguntas interactivas.

Las interfaces anteriores `download_raw()` y `download()` siguen disponibles para los procedimientos que admiten.

Para utilizar la configuración JSON actual y su política de fechas, debe emplearse `download_cob()`.

## Verificación y análisis posteriores

### Pruebas del proyecto

Las pruebas unitarias y de regresión que se ejecutan sin conexión utilizan respuestas simuladas y carpetas temporales.

Estas pruebas no comprueban las suscripciones reales a los proveedores ni realizan solicitudes a sus API.

Para ejecutar las pruebas, utiliza el siguiente comando:

```bash
python -m unittest discover -s tests -v
```

Las nuevas pruebas relacionadas con COB comprueban:

- La validación de las fechas.
- Las dos modalidades de consulta de opciones: histórica y actual.
- La reutilización de datos descargados previamente.
- La recuperación de ejecuciones interrumpidas.
- La detección de configuraciones incompatibles.
- El tratamiento de campos incompletos del modelo.

### Instalación del proyecto

El proyecto requiere **Python 3.11 o una versión posterior**.

Para instalarlo junto con sus dependencias declaradas, ejecuta:

```bash
python -m pip install -e .
```

Entre las dependencias se incluye `ib_insync`, utilizado para la comunicación con Interactive Brokers.

El SDK oficial independiente `ibapi` ya no se utiliza.

Para importar la biblioteca, emplea:

```python
from src import MarketDataDownloader
```

No añadas directamente la carpeta `src` a la variable `PYTHONPATH`.

El motivo es que el archivo `src/http.py` podría entrar en conflicto con un módulo de la biblioteca estándar de Python.

### Funcionalidades implementadas y pendientes

Actualmente están implementadas las siguientes funcionalidades:

- Limpieza de datos.
- Alineación de rentabilidades en moneda local.
- Estadísticas descriptivas.
- Simulaciones históricas de Monte Carlo.

Monte Carlo también puede simular conjuntamente los factores de tipos de cambio necesarios cuando existe un historial adecuado.

Quedan pendientes para fases posteriores:

- Previsiones de dividendos.
- Construcción de curvas de descuento y tipos a plazo (*forward curves*).
- Superficies de volatilidad implícita.
- Simulaciones basadas en volatilidad local.

Las observaciones de ESTR y SOFR y los rendimientos de deuda soberana no sustituyen a una curva de descuento OIS.

El tipo de cambio EUR/USD expresa **cuántos dólares estadounidenses equivalen a un euro**.

Por tanto, para convertir importes a euros es necesario disponer de la observación del tipo de cambio correspondiente a la fecha analizada.

### Documentación de IBKR

Se pueden consultar las siguientes referencias técnicas de Interactive Brokers:

- [Barras históricas de IBKR](https://interactivebrokers.github.io/tws-api/historical_bars.html).
- [Limitaciones de los datos históricos de IBKR](https://interactivebrokers.github.io/tws-api/historical_limitations.html).

### Identificación de resultados en la consola

Los resultados mostrados en la consola distinguen entre:

- `[HISTORICAL]`: datos históricos.
- `[LIVE SNAPSHOT]`: cotizaciones actuales capturadas en un momento determinado.

Los identificadores de solicitudes existentes que terminan en `_HISTORY` se mantienen sin cambios para permitir la reutilización de archivos anteriores.

Sin embargo, ese sufijo no determina qué procedimiento de consulta se ejecuta.

Los nuevos metadatos de resultados registran la operación realmente ejecutada (`operation`) por separado de la operación configurada (`configured_operation`).