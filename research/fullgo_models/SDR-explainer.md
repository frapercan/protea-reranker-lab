# Representaciones dispersas (SDR) para transferencia de función proteica: qué probamos y qué encontramos

**Resumen en una línea.** Probamos si una representación *dispersa* de proteínas predice la
similitud funcional (GO) mejor que la *densa*. La dispersa naive pierde, pero descomponiendo el
experimento descubrimos que **el problema no es la dispersión (es casi gratis): es la binarización**
(tirar las magnitudes). Eso reorienta el trabajo: una dispersa con valores reales ya iguala a la
densa, y para una dispersa *binaria* hace falta aprender un código alineado con la función.

---

## 1. El contexto y la hipótesis

Un modelo de lenguaje de proteínas (PLM, aquí **ankh-base**) convierte una secuencia de aminoacidos
en un vector **denso** de 768 numeros reales (el *embedding*). Ese vector codifica propiedades de la
proteina; proteinas funcionalmente parecidas tienden a quedar cerca en ese espacio.

La hipotesis (del documento `sparse.pdf`) es que una representacion **dispersa** (SDR, *Sparse
Distributed Representation*: un vector donde casi todas las coordenadas son cero y solo unas pocas
estan activas) podria capturar la **funcion** mejor que la densa, porque concentraria la senal en
las dimensiones que importan y descartaria el ruido.

La forma mas simple de obtener un SDR a partir del denso es **k-WTA** (*k-Winners-Take-All*):
quedarse con las **k coordenadas de mayor magnitud** y poner el resto a cero. Para comparar dos SDR
binarios se usa **Tanimoto** (solapamiento de conjuntos activos: `interseccion / union`).

**Pregunta del experimento:** la similitud Tanimoto entre SDR, ¿correlaciona con la similitud
funcional real al menos tan bien como el coseno entre los vectores densos?

---

## 2. Como se mide (el banco de pruebas)

Es una medida **intrinseca** de la representacion (no una prediccion):

1. **Datos.** Pool de referencia v227 (anotaciones GO conocidas a fecha de corte, *leakage-clean*),
   5000 proteinas muestreadas, 200000 pares de proteinas (semilla 42).
2. **Verdad de terreno (la funcion).** Para cada par de proteinas se calcula su **similitud
   semantica GO** sobre el grafo de la ontologia: **Resnik** (informacion compartida del ancestro
   comun mas informativo) y **Lin** (la misma, normalizada a [0,1]). Son las dos metricas estandar.
3. **Similitud de representacion.** Para cada par: coseno (denso) y Tanimoto (disperso).
4. **La medida.** **Correlacion de Spearman** (de rangos) entre la similitud de representacion y la
   similitud GO, sobre los 200000 pares. Mayor Spearman = la geometria de la representacion sigue
   mejor a la funcion. La pregunta exacta: *¿que similitud ordena los pares mas parecido a como los
   ordena la funcion?*

---

## 3. Los experimentos, en orden

### 3.1. SDR naive (k-WTA sobre el vector medio): NEGATIVO
Primer intento: k-WTA sobre el embedding mean-pooled. Resultado (ProtT5): coseno denso **0.315** vs
Tanimoto disperso **0.255** (Resnik). El disperso pierde por ~0.06.

### 3.2. El re-test justo (dispersar-y-luego-agregar, por trozos): SIGUE NEGATIVO
Objecion legitima: ambos brazos partian del **mismo vector promediado**, asi que quiza el culpable
era el *promediado*, no la dispersion. El orden correcto (segun `sparse.pdf`) es **dispersar cada
trozo y luego agregar** (*sparsify-then-bundle*), no al reves. Para probarlo embebimos las proteinas
**por trozos** (ankh-base, chunks de 512) y construimos el SDR de proteina como el *bundle* de los
k-WTA de sus trozos.

Resultado: **identico** al naive (a dos decimales), y los dos por debajo del denso. **El orden de la
agregacion no cambia nada.** Ademas lo repetimos sobre la ventana 227-230 (v230): mismo veredicto
(la geometria es agnostica a la ventana, son los mismos vectores). Conclusion provisional: la
familia "dispersar por magnitud" es genuinamente mas debil. **Pero todavia no sabiamos POR QUE.**

### 3.3. La descomposicion (el experimento clave): aislar las tres variables
La comparacion "denso coseno vs disperso Tanimoto" mezcla **tres cambios a la vez**: (1) quedarse
con pocas dimensiones (dispersar), (2) tirar las magnitudes de las que quedan (binarizar), (3)
cambiar la metrica (coseno -> Tanimoto). El gap podia venir de cualquiera. Construimos una **escalera
de 4 peldanos** que los aisla, sobre el mismo vector:

| | dimensiones | valores | metrica | que aisla |
|---|---|---|---|---|
| **A** | 768 (todas) | reales | coseno | denso (linea base) |
| **B** | top-k | **reales** | coseno | + solo dispersar |
| **C** | top-k | binario | coseno | + binarizar |
| **D** | top-k | binario | Tanimoto | lo que haciamos |

Detalle tecnico: a k fijo, coseno-binario y Tanimoto son ambos monotonos en el solape, asi que dan
**el mismo Spearman** (C = D). Por eso la metrica no es la variable; lo que de verdad se mide es
A->B (dispersar) y B->C (binarizar).

---

## 4. Resultado (ver la figura `sdr_decomposition.png`)

Spearman vs Resnik (ankh-base, v227):

| peldano | k=32 | k=64 | k=128 |
|---|---|---|---|
| **A** denso (full, real, coseno) | 0.2315 | 0.2315 | 0.2315 |
| **B** disperso REAL (top-k + magnitudes) | 0.2111 | 0.2175 | **0.2203** |
| **C** disperso BINARIO (top-k, 1/0, Tanimoto) | 0.1300 | 0.1052 | 0.0934 |
| **coste de DISPERSAR (A->B)** | -0.020 | -0.014 | **-0.011** |
| **coste de BINARIZAR (B->C)** | -0.081 | -0.112 | **-0.127** |

(Lin da el mismo patron: dispersar ~ -0.013, binarizar ~ -0.09 a -0.12.)

**Lectura:**
- **Dispersar es casi gratis.** Quedarse con 128 de 768 dimensiones (con sus valores reales) cuesta
  solo **-0.011**. La informacion densa vive en pocas dimensiones.
- **Binarizar es el que mata:** **-0.13**, entre 5x y 10x mas costoso que dispersar. Todo el gap
  denso-vs-disperso era, en realidad, **el coste de tirar las magnitudes.**
- **La tendencia en k se invierte y lo confirma:** en B (real) mas k = mejor (sube hacia el denso);
  en C (binario) mas k = peor (metes dimensiones de baja magnitud, ruido, puestas a 1). Por eso el
  binario prefiere k pequeno: cuando tiras la magnitud, solo los bits mas fuertes son fiables.

---

## 5. Que significa (y por que importa)

1. **La idea dispersa es viable.** Un codigo **disperso de valores reales** (top-k + magnitudes,
   coseno) practicamente **iguala al denso** (-0.01). Si solo quieres compresion/dispersion, ya la
   tienes casi gratis, sin aprender nada.
2. **El binario naive no funciona** porque la magnitud llevaba la senal funcional.
3. **Reformula el problema de "alinear con la tarea":** alinear no es para la dispersion (esa es
   gratis); es para que **el patron de activacion binario en si mismo signifique funcion**, de modo
   que no necesites las magnitudes. Ese es el trabajo de un codigo **aprendido** (un autoencoder
   disperso con objetivo de funcion / contrastivo): aprender *que* bits encender para que el patron,
   sin pesos, ya discrimine funcion.

**En una frase:** no es "disperso pierde", es "binarizar pierde"; dispersar es gratis. El siguiente
paso con sentido es un codigo binario **aprendido y alineado con la funcion**, no la k-WTA naive.

---

## 6. Por que este resultado es solido

- **Leakage-clean:** la verdad de terreno usa solo anotaciones t0 (v227); el embedding es agnostico
  a la ventana (verificado: v227 y v230 dan el mismo veredicto).
- **Controlado:** la escalera aisla cada variable sobre el mismo vector y el mismo conjunto de pares;
  el unico cambio entre peldanos es el que se quiere medir.
- **Honesto:** es un **resultado negativo** para la hipotesis simple, pero util: descarta una via
  barata y senala con precision donde esta el cuello (la binarizacion) y por tanto que debe atacar
  el metodo aprendido. Cada metrica (Resnik y Lin) da el mismo signo.
- **Reproducible:** todo trazado en MLflow; el script y los datos quedan versionados.

---

## Apendice: definiciones exactas

- **Embedding (denso):** ankh-base, 768-dim, real. Por proteina = media de los vectores por trozo.
- **k-WTA(v, k):** indices de las k coordenadas de mayor `|v_i|`; el resto a cero.
- **SDR binario:** vector 1/0 con esos k indices a 1.
- **Tanimoto(x, y)** (binarios, k bits cada uno): `i / (2k - i)`, con `i` = bits en comun.
- **Coseno(x, y):** `<x,y> / (||x|| ||y||)`.
- **Information Content** `IC(t) = -log p(t)`, p(t) = frecuencia del termino t (y descendientes) en
  el corpus muestreado propagado por el DAG.
- **Resnik(P, Q):** IC del ancestro comun mas informativo entre los conjuntos de terminos de P y Q.
- **Lin(P, Q):** `2 * IC(MICA) / (IC_best_P + IC_best_Q)`, en [0,1].
- **Datos:** pool v227 (annotation set `c905dffa`), 5000 proteinas, 200000 pares, semilla 42.
  ankh-base chunked cs=512 (config `6542db1e`), corpus completo embebido.

*Nota: los valores absolutos de Spearman dependen del PLM (ankh-base ~0.22 Resnik; ProtT5 ~0.31 en el
experimento original); ankh-base es un proxy de GO algo mas debil aqui. La cantidad comparable es el
**gap relativo** entre peldanos, que es robusto y reproducible.*
