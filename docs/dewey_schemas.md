## Dewey Schemas

This file documents the Dewey inputs used by the replication package without redistributing the underlying proprietary records.

### Global Places

Local coverage summary:

- Local file count: `32`
- Example file: `data/raw/dewey/global_places/california-global-places-data/global-places-poi-geometry_0_0_0.snappy.parquet`
- Purpose in pipeline: POI geometry, category labels, and identifier crosswalks

Observed columns from a local file:

`BRANDS`, `CATEGORY_TAGS`, `CITY`, `CLOSED_ON`, `DOMAINS`, `ENCLOSED`, `GEOMETRY_TYPE`, `INCLUDES_PARKING_LOT`, `ISO_COUNTRY_CODE`, `IS_SYNTHETIC`, `LATITUDE`, `LOCATION_NAME`, `LONGITUDE`, `NAICS_CODE`, `NAICS_CODE_2022`, `OPENED_ON`, `OPEN_HOURS`, `PARENT_PLACEKEY`, `PHONE_NUMBER`, `PLACEKEY`, `POLYGON_CLASS`, `POLYGON_WKT`, `POSTAL_CODE`, `REGION`, `STORE_ID`, `STREET_ADDRESS`, `SUB_CATEGORY`, `SUB_CATEGORY_2022`, `TOP_CATEGORY`, `TOP_CATEGORY_2022`, `TRACKING_CLOSED_SINCE`, `WEBSITE`, `WKT_AREA_SQ_METERS`

Key fields used by the replication code:

- `PLACEKEY`
- `STORE_ID`
- `LOCATION_NAME`
- `LATITUDE`
- `LONGITUDE`
- `NAICS_CODE`
- `OPENED_ON`
- `REGION`

### Spend Patterns

Local coverage summary:

- Local file count: `427`
- Local partition range: `2019-01` to `2026-01`
- Example file: `data/raw/dewey/spend_patterns/spend/2019-01--data_01c05993-0107-0032-0042-fa07042edf46_209_0_0.snappy.parquet`
- Purpose in pipeline: monthly customer counts, spending, and income-bucket outcomes

Observed columns from a local file:

`BRANDS`, `BUCKETED_CUSTOMER_FREQUENCY`, `BUCKETED_CUSTOMER_INCOMES`, `CITY`, `CUSTOMER_HOME_CITY`, `DAY_COUNTS`, `ISO_COUNTRY_CODE`, `LATITUDE`, `LOCATION_NAME`, `LONGITUDE`, `MEAN_SPEND_PER_CUSTOMER_BY_FREQUENCY`, `MEAN_SPEND_PER_CUSTOMER_BY_INCOME`, `MEDIAN_SPEND_PER_CUSTOMER`, `MEDIAN_SPEND_PER_TRANSACTION`, `NAICS_CODE`, `ONLINE_SPEND`, `ONLINE_TRANSACTIONS`, `PARENT_PLACEKEY`, `PLACEKEY`, `POSTAL_CODE`, `RAW_NUM_CUSTOMERS`, `RAW_NUM_TRANSACTIONS`, `RAW_TOTAL_SPEND`, `REGION`, `RELATED_BUYNOWPAYLATER_SERVICE_PCT`, `RELATED_CROSS_SHOPPING_LOCAL_BRANDS_PCT`, `RELATED_CROSS_SHOPPING_ONLINE_MERCHANTS_PCT`, `RELATED_CROSS_SHOPPING_PHYSICAL_BRANDS_PCT`, `RELATED_CROSS_SHOPPING_SAME_CATEGORY_BRANDS_PCT`, `RELATED_DELIVERY_SERVICE_PCT`, `RELATED_PAYMENT_PLATFORM_PCT`, `RELATED_RIDESHARE_SERVICE_PCT`, `RELATED_STREAMING_CABLE_PCT`, `RELATED_WIRELESS_CARRIER_PCT`, `SPEND_BY_DAY`, `SPEND_BY_DAY_OF_WEEK`, `SPEND_BY_TRANSACTION_INTERMEDIARY`, `SPEND_DATE_RANGE_END`, `SPEND_DATE_RANGE_START`, `SPEND_PCT_CHANGE_VS_PREV_MONTH`, `SPEND_PCT_CHANGE_VS_PREV_YEAR`, `SPEND_PER_TRANSACTION_BY_DAY`, `SPEND_PER_TRANSACTION_PERCENTILES`, `STREET_ADDRESS`, `SUB_CATEGORY`, `TOP_CATEGORY`, `TRANSACTION_INTERMEDIARY`

Key fields used by the replication code:

- `PLACEKEY`
- `SPEND_DATE_RANGE_START`
- `RAW_NUM_CUSTOMERS`
- `RAW_TOTAL_SPEND`
- `MEDIAN_SPEND_PER_TRANSACTION`
- `BUCKETED_CUSTOMER_INCOMES`

### Foot Traffic

Local coverage summary:

- Local file count at packaging time: `0`
- Purpose in pipeline: monthly dwell-time and distance-from-home outcomes
- Acquisition path: Dewey weekly API download through [code/00_data_download/01_download_dewey_foot_traffic.py](../code/00_data_download/01_download_dewey_foot_traffic.py)

Observed columns from an inspected weekly sample file:

`BRAND`, `BUCKETED_DWELL_TIMES`, `CITY`, `CLOSE_DATE`, `DATE_RANGE_END`, `DATE_RANGE_START`, `DEVICE_TYPE`, `DISTANCE_FROM_HOME`, `FOOTPRINT_ID`, `ID_STORE`, `ISO_COUNTRY_CODE`, `IS_DISTRIBUTOR`, `LATITUDE`, `LOCATION_NAME`, `LONGITUDE`, `MEDIAN_DWELL`, `MSA_CODE`, `NAICS_CODE`, `OPEN_DATE`, `PERSISTENT_ID`, `PERSISTENT_ID_STORE`, `POI_CBG`, `POSTAL_CODE`, `REGION`, `RELATED_SAME_DAY_BRAND`, `RELATED_SAME_WEEK_BRAND`, `STREET_ADDRESS`, `SUB_CATEGORY`, `TICKER`, `TOP_CATEGORY`, `VISITOR_COUNTRY_OF_ORIGIN`, `VISITOR_COUNTS`, `VISITOR_DAYTIME_CBGS`, `VISITOR_HOME_AGGREGATION`, `VISITOR_HOME_CBGS`, `VISITS_BY_DAY`, `VISITS_BY_EACH_HOUR`, `VISIT_COUNTS`

Key fields used by the replication code:

- `DATE_RANGE_START`
- `MEDIAN_DWELL`
- `DISTANCE_FROM_HOME`
- `ID_STORE`
- `LATITUDE`
- `LONGITUDE`
- `LOCATION_NAME`

Important note:

The weekly foot-traffic files do not expose `PLACEKEY` directly in the inspected sample. The Python pipeline therefore joins them back to the POI universe through Dewey global-places identifier metadata and fallback location matching.
