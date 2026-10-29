# Guía en español — Taller B1 T1
[← Volver al README principal](README.md)

## 1. ¿En qué consiste nuestro proyecto?

El proyecto desarrolla un programa en Python para obtener, organizar y analizar datos financieros, estudiar una cartera de inversión y realizar simulaciones de posibles escenarios futuros.

El objetivo es comprender cómo puede comportarse una cartera en diferentes situaciones de mercado y analizar sus riesgos.

## 2. ¿Qué cartera estamos estudiando?

La configuración actual incluye dos fondos cotizados (ETF):

- **SPY (50 %):** ETF que sigue el índice S&P 500, compuesto por grandes empresas estadounidenses.
- **IEF (50 %):** ETF de bonos del Tesoro de Estados Unidos con vencimientos aproximados de 7 a 10 años.

La cartera tiene una distribución del 50 % en renta variable y del 50 % en renta fija soberana.

La moneda de referencia del análisis es el euro (EUR).

## 3. ¿Cómo funciona el programa?

El proceso se organiza en varias etapas:

1. **Obtención de datos:** consulta información financiera de distintos proveedores.
2. **Comprobación:** revisa los datos disponibles y detecta posibles problemas.
3. **Limpieza:** prepara los datos para que puedan analizarse correctamente.
4. **Análisis estadístico:** estudia el comportamiento histórico de los activos.
5. **Simulación Monte Carlo:** genera numerosos escenarios posibles de evolución de la cartera.
6. **Resultados:** prepara informes, estadísticas y gráficos para facilitar su interpretación.

## 4. ¿Qué es una simulación Monte Carlo?

Es una técnica que permite estudiar muchos escenarios posibles en lugar de intentar predecir un único resultado.

En la configuración actual del proyecto se establecen **10.000 trayectorias simuladas**.

También se utilizan cuantiles y niveles de confianza del 95 % y 99 % para estudiar la distribución de los resultados y sus riesgos.

**Importante:** una simulación no garantiza lo que ocurrirá realmente en los mercados.

## 5. ¿Qué hemos comprobado?

Se han ejecutado correctamente las siete pruebas del archivo `tests/test_portfolio.py`, incluida una prueba añadida para comprobar que el programa rechaza un valor negativo de la cartera.

Estas pruebas verifican determinadas reglas de validación de la cartera, pero no demuestran por sí solas que todo el programa funcione correctamente en cualquier situación.

## 6. Glosario básico

- **Python:** lenguaje de programación utilizado para desarrollar el proyecto.
- **ETF:** fondo de inversión que cotiza en bolsa.
- **S&P 500:** índice de referencia de grandes empresas estadounidenses.
- **Cartera:** conjunto de inversiones.
- **Proveedor de datos:** servicio del que se obtiene información financiera.
- **Monte Carlo:** método para simular numerosos escenarios posibles.
- **Cuantil:** valor que marca una posición determinada dentro de una distribución de resultados.
- **GitHub:** plataforma que permite compartir código y colaborar en un proyecto.

