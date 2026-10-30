# Guía en español — Taller B1 T1
[← Volver al README principal](README.md)

## 1. ¿En qué consiste nuestro proyecto?

El proyecto desarrolla un programa en Python para obtener, organizar, procesar y analizar datos de mercados financieros.

Permite estudiar el comportamiento histórico de distintos activos, analizar carteras de inversión y realizar simulaciones Monte Carlo para explorar posibles escenarios futuros.

El objetivo es comprender cómo pueden comportarse las inversiones bajo diferentes condiciones de mercado y evaluar los riesgos asociados.

## 2. ¿Qué cartera podemos estudiar?

El programa permite analizar una cartera definida mediante archivos de configuración.

Una cartera puede estar formada por distintos activos financieros, como fondos cotizados (ETF), con pesos que determinan la proporción invertida en cada uno.

Su composición depende de la configuración utilizada en cada ejecución.

El análisis también puede considerar la moneda de referencia de la cartera y, cuando corresponde, los tipos de cambio necesarios.

## 3. ¿Cómo funciona el programa?

El proceso se organiza en varias etapas:

1. **Configuración:** se definen los parámetros del análisis, como activos, fechas y otras opciones.
2. **Obtención de datos:** se consulta información financiera de los proveedores disponibles.
3. **Comprobación:** se revisa la información recibida y se identifican posibles incidencias.
4. **Limpieza y procesamiento:** se preparan los datos para que puedan analizarse correctamente.
5. **Análisis estadístico:** se estudia el comportamiento histórico de los activos.
6. **Simulación Monte Carlo:** se generan escenarios posibles de evolución de la cartera.
7. **Resultados:** se producen informes, estadísticas y gráficos que facilitan su interpretación.

## 4. ¿Cómo se ejecuta?

Una vez preparado el entorno de Python e instaladas las dependencias del proyecto, se puede iniciar el programa desde la carpeta principal del repositorio:

```powershell
python run.py
```

El programa permite trabajar con datos previamente almacenados o solicitar nuevos datos a los proveedores disponibles.

Las opciones concretas de ejecución y los parámetros utilizados deben consultarse en la documentación principal y en los archivos de configuración.

La disponibilidad de datos puede depender de las credenciales, los proveedores externos y las fechas seleccionadas.

## 5. ¿Qué es una simulación Monte Carlo?

Monte Carlo es una técnica matemática y estadística que permite estudiar numerosos escenarios posibles en lugar de intentar predecir un único resultado.

Para realizar las simulaciones, el programa utiliza parámetros estimados a partir de información histórica, como:

- **Rentabilidad media:** comportamiento medio observado de los rendimientos.
- **Volatilidad:** variabilidad de esos rendimientos.
- **Covarianza y correlación:** relaciones entre los movimientos de distintos activos o factores financieros.

A partir de estos parámetros se generan trayectorias simuladas de evolución de la cartera.

El número de simulaciones, los modelos utilizados y el horizonte temporal dependen de la configuración seleccionada.

**Importante:** los resultados simulados no son predicciones garantizadas ni recomendaciones de inversión.

## 6. ¿Cómo se analiza el riesgo?

El programa permite estudiar los resultados de las simulaciones mediante diferentes indicadores estadísticos y de riesgo.

Entre ellos se encuentran:

- **Media:** resultado promedio de las simulaciones.
- **Mediana:** valor que divide los resultados ordenados en dos mitades.
- **Cuantiles:** valores que permiten analizar diferentes partes de la distribución.
- **Probabilidad de pérdida:** proporción de escenarios simulados que terminan por debajo del valor inicial.
- **Value at Risk (VaR):** umbral de pérdida asociado a un nivel de confianza y un horizonte determinados.
- **Expected Shortfall (ES):** pérdida media en los escenarios más desfavorables que superan el umbral VaR.
- **Máximo drawdown:** mayor caída desde un máximo anterior durante la evolución de una inversión.

Estos indicadores permiten estudiar tanto los resultados centrales como las posibles situaciones adversas.

## 7. ¿Cómo se comprueba el funcionamiento?

El proyecto incorpora pruebas automáticas destinadas a verificar determinados comportamientos del programa.

Estas pruebas pueden comprobar, entre otros aspectos:

- La validación de los datos de entrada.
- Las reglas de configuración de las carteras.
- El tratamiento de situaciones incorrectas o inesperadas.
- El funcionamiento de componentes específicos del código.

Además de las pruebas automáticas, el programa puede ejecutarse para comprobar su funcionamiento con los datos y proveedores disponibles.

Es importante distinguir entre:

- **Pruebas automáticas:** verifican comportamientos concretos del código.
- **Pruebas de ejecución:** permiten observar el funcionamiento del programa en determinadas condiciones.
- **Resultados financieros:** dependen de los datos, fechas, activos y modelos seleccionados.

Superar las pruebas automáticas no garantiza que todos los proveedores externos estén siempre disponibles ni que el programa funcione correctamente en cualquier situación.

## 8. ¿Qué resultados genera?

Según la configuración y los datos disponibles, el programa puede generar distintos archivos de salida:

- **Informes en Markdown:** presentan los resultados y las características del análisis.
- **Archivos CSV:** contienen tablas con parámetros, estadísticas y medidas de riesgo.
- **Gráficos PNG:** facilitan la visualización de trayectorias simuladas y resultados financieros.

Los archivos se organizan dentro de la estructura de resultados del proyecto, permitiendo identificar las distintas ejecuciones.

Cada resultado debe interpretarse teniendo en cuenta la configuración y el historial utilizados.

## 9. ¿Qué limitaciones debemos tener en cuenta?

Como ocurre con cualquier herramienta de análisis financiero, existen determinadas limitaciones:

- La disponibilidad y calidad de los datos dependen de las fuentes consultadas.
- Algunos proveedores pueden requerir credenciales o permisos.
- Los resultados estadísticos dependen del período histórico utilizado.
- Las simulaciones utilizan hipótesis matemáticas que pueden no reproducir todas las situaciones reales del mercado.
- No todas las funcionalidades contempladas en un proyecto tienen necesariamente el mismo grado de desarrollo o validación.

Por ello, conviene revisar los informes de cada ejecución y consultar la documentación técnica para conocer el alcance de los resultados.

## 10. Glosario básico

- **Python:** lenguaje de programación utilizado para desarrollar el proyecto.
- **ETF:** fondo de inversión que cotiza en bolsa.
- **Cartera:** conjunto de inversiones financieras.
- **Proveedor de datos:** servicio que suministra información financiera.
- **COB (Close of Business):** fecha de referencia de cierre utilizada en el procesamiento.
- **Monte Carlo:** método para generar y estudiar numerosos escenarios simulados.
- **Rentabilidad:** variación del valor de una inversión.
- **Volatilidad:** medida de la variabilidad de los rendimientos.
- **Correlación:** medida de la relación estadística entre los movimientos de dos variables.
- **Cuantil:** valor que señala una posición dentro de una distribución.
- **VaR:** medida del umbral de pérdida para un nivel de confianza y horizonte determinados.
- **Expected Shortfall:** pérdida media en los escenarios que superan el umbral VaR.
- **Drawdown:** caída desde un máximo anterior del valor de una inversión.
- **GitHub:** plataforma para compartir código y colaborar en proyectos.

## 11. Conclusión

Este proyecto integra diferentes etapas del análisis financiero: obtención de datos, procesamiento, análisis estadístico y simulación de escenarios.

Su estructura permite estudiar carteras con diferentes configuraciones y analizar tanto los resultados esperados como los riesgos asociados.

La documentación, las pruebas y los informes generados facilitan la comprensión, revisión y evolución del proyecto de forma colaborativa.