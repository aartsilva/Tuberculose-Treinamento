# Dicionário de Dados Simplificado

Este projeto usa dados do SINAN/DataSUS sobre tuberculose.

## Variável alvo

| Coluna | Significado |
|---|---|
| `ltfu` | Indica abandono do tratamento. 0 = não abandonou, 1 = abandonou. |

## Preditores numéricos

| Coluna | Uso |
|---|---|
| `idade_anos` | Idade do paciente em anos. |
| `NU_ANO` | Ano da notificação. |
| `NU_CONTATO` | Número de contatos registrados. |

## Preditores categóricos

| Coluna | Uso |
|---|---|
| `SG_UF_NOT` | UF de notificação. |
| `CS_SEXO` | Sexo. |
| `CS_GESTANT` | Gestante. |
| `CS_RACA` | Raça/cor. |
| `CS_ESCOL_N` | Escolaridade. |
| `TRATAMENTO` | Tipo de tratamento. |
| `INSTITUCIO` | Institucionalização. |
| `RAIOX_TORA` | Resultado/realização de raio X de tórax. |
| `TESTE_TUBE` | Teste tuberculínico. |
| `FORMA` | Forma clínica. |
| `AGRAVAIDS` | Agravo AIDS. |
| `AGRAVALCOO` | Agravo alcoolismo. |
| `AGRAVDIABE` | Agravo diabetes. |
| `AGRAVDOENC` | Agravo doença mental. |
| `AGRAVOUTRA` | Outros agravos. |
| `BACILOSC_E` | Baciloscopia de escarro. |
| `BACILOS_E2` | Segunda baciloscopia. |
| `BACILOSC_O` | Outra baciloscopia. |
| `CULTURA_ES` | Cultura de escarro. |
| `CULTURA_OU` | Outra cultura. |
| `HIV` | Teste HIV. |
| `HISTOPATOL` | Histopatologia. |
| `TRAT_SUPER` | Tratamento supervisionado. |
| `DOENCA_TRA` | Doença relacionada ao trabalho. |
| `POP_LIBER` | População privada de liberdade. |
| `POP_RUA` | População em situação de rua. |
| `POP_SAUDE` | Profissional de saúde. |
| `POP_IMIG` | Imigrante. |
| `BENEF_GOV` | Beneficiário de programa governamental. |
| `AGRAVDROGA` | Uso de drogas. |
| `AGRAVTABAC` | Tabagismo. |
| `TEST_MOLEC` | Teste molecular. |
| `TEST_SENSI` | Teste de sensibilidade. |
| `ANT_RETRO` | Terapia antirretroviral. |

## Colunas removidas por data leakage

| Coluna | Motivo |
|---|---|
| `SITUA_ENCE` | Situação de encerramento do caso. |
| `DT_ENCERRA` | Data de encerramento. |
| `SITUA_9_M` | Situação após 9 meses. |
| `SITUA_12_M` | Situação após 12 meses. |
| `TRANSF` | Informação de transferência. |
| `UF_TRANSF` | UF de transferência. |
| `MUN_TRANSF` | Município de transferência. |
| `BAC_APOS_6` | Informação posterior ao início do tratamento. |
