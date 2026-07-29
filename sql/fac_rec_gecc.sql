WITH FAC AS (
    SELECT
        f.strPeriodo        AS "Periodo",
        f.strIdentificacion AS "Id",
        SUM(f.numValorVencido)        AS "Valor_Vencido",
        SUM(f.numValorTotalCalculado) AS "Valor_Cuota_Mes"
    FROM BodegaCorporativa.bodega.factFacturaEstadoCuentaGECC f
    INNER JOIN BodegaCorporativa.bodega.dimConceptoFacturaGECC c
        ON f.strCodConcepto = c.strCodConcepto
    WHERE
        BodegaCorporativa.$partition.pf_mes(f.dtmFechaInsercion) = BodegaCorporativa.$partition.pf_mes('?')
        AND f.strIdentificacion IN ({ids})
    GROUP BY f.strPeriodo, f.strIdentificacion
),
REC AS (
    SELECT
        r.strPeriodo        AS "Periodo",
        r.strIdentificacion AS "Id",
        SUM(r.numRecaudoVencido) AS "Recaudo_Vencido",
        SUM(r.numRecaudoTotal)   AS "RecaudoTotal"
    FROM BodegaCorporativa.bodega.factRecaudoMesConceptoGECC r
    INNER JOIN BodegaCorporativa.bodega.dimConceptoFacturaGECC c
        ON r.strCodConcepto = c.strCodConcepto
    WHERE
        BodegaCorporativa.$partition.pf_mes(r.dtmFechaInsercion) = BodegaCorporativa.$partition.pf_mes('?')
        AND r.strIdentificacion IN ({ids})
    GROUP BY r.strPeriodo, r.strIdentificacion
)
SELECT
    COALESCE(FAC."Periodo", REC."Periodo") AS "Periodo",
    COALESCE(FAC."Id",      REC."Id")      AS "Id",
    FAC."Valor_Vencido",
    FAC."Valor_Cuota_Mes",
    REC."Recaudo_Vencido",
    REC."RecaudoTotal"
FROM FAC
FULL JOIN REC
    ON FAC."Periodo" = REC."Periodo"
    AND FAC."Id"     = REC."Id";