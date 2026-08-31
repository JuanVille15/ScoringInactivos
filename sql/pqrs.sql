WITH CategoriasHomologadas AS (
	SELECT
		NRO_DOCUMENTO,
		CASE 
			WHEN CATEGORIA IN ('Solicitud','Petición','Petici¢n','Derecho de Petición','Solicitud Ente de control') THEN 'Peticiones y Solicitudes'
			WHEN CATEGORIA IN ('Reclamo','Queja','Reclamos Riesgo Simple','Reclamo Riesgo Priorizado','Reclamo Riesgo Vital','PQR Riesgo de Vida') THEN 'Quejas y Reclamos'
			WHEN CATEGORIA IN ('Entes de Control','Requerimientos Gubernamentales Y Judiciales') THEN 'Requerimientos Externos'
		ELSE 'Otros tipos de PQRS'
		END AS GrupoPQRS,
	FEC_REGISTRO
	FROM [StageCorporativa].[Stage].[extrCRMPQRs]
	WHERE 
		NRO_DOCUMENTO IN ({bloque_str})
		AND CATEGORIA <> 'No Tiene / No Aplica'
		AND FEC_REGISTRO BETWEEN ? AND ?
)
SELECT
	NRO_DOCUMENTO AS [Identificacion],
	Count(*) AS [Cantidad_pqr_ultimo_anno]
FROM CategoriasHomologadas
WHERE
	GrupoPQRS IN ('Peticiones y Solicitudes', 'Quejas y Reclamos')
GROUP BY 
	NRO_DOCUMENTO; 