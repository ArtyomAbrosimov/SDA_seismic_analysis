import h5py
import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

NOISE_BUFFER_SEC = 600
EVENT_DURATION_SEC = 120
TOTAL_DURATION = NOISE_BUFFER_SEC + EVENT_DURATION_SEC
LIMIT_COUNT = 50

PROVIDER = "INGV"
NETWORK = "IV"
MIN_MAGNITUDE = 2
SEARCH_RADIUS_DEG = 5
CENTER_LAT = 42.0
CENTER_LON = 13.0

START_DATE = UTCDateTime("2019-01-01")
END_DATE = UTCDateTime("2020-01-01")

OUTPUT_H5 = "Instance_Auto_Events.hdf5"
OUTPUT_CSV = "metadata_auto_events.csv"

client = Client(PROVIDER)

catalog = client.get_events(
    starttime=START_DATE,
    endtime=END_DATE,
    minmagnitude=MIN_MAGNITUDE,
    latitude=CENTER_LAT,
    longitude=CENTER_LON,
    maxradius=SEARCH_RADIUS_DEG,
    limit=LIMIT_COUNT
)
print(f"Найдено событий: {len(catalog)}")

with h5py.File(OUTPUT_H5, 'w') as f:
    data_group = f.create_group("data")
    metadata_list = []
    saved_count = 0

    for event in catalog:
        origin = event.origins[0]
        mag = event.magnitudes[0].mag
        otime = origin.time
        ev_lat = origin.latitude
        ev_lon = origin.longitude

        inventory = client.get_stations(
            network=NETWORK,
            latitude=ev_lat,
            longitude=ev_lon,
            maxradius=0.5,
            starttime=otime,
            endtime=otime + 100,
            level="station"
        )

        stations_found = [net[0].code for net in inventory for sta in net]
        downloaded_event = False

        for station_code in stations_found:
            if downloaded_event:
                break

            t_start = otime - NOISE_BUFFER_SEC
            t_end = t_start + TOTAL_DURATION

            try:
                st = client.get_waveforms(NETWORK, station_code, "*", "HH*,BH*,EH*", t_start, t_end)
            except Exception:
                continue

            st.merge(fill_value=0)
            chan_codes = set([tr.stats.channel[:2] for tr in st])
            final_st = None

            for code in chan_codes:
                sub = st.select(channel=f"{code}*")
                if len(sub) >= 3:
                    final_st = sub
                    break

            if not final_st:
                continue

            final_st.sort()
            final_st.resample(100.0)

            final_st.trim(t_start, t_start + TOTAL_DURATION)
            n_pts = min([tr.stats.npts for tr in final_st])
            if n_pts < (TOTAL_DURATION * 95):
                continue

            data_np = np.array([tr.data[:n_pts] for tr in final_st])
            trace_name = f"{NETWORK}.{station_code}.{otime.strftime('%Y%m%d%H%M%S')}"
            data_group.create_dataset(trace_name, data=data_np)
            arrival_sample = int(NOISE_BUFFER_SEC * 100.0)

            metadata_list.append({
                'trace_name': trace_name,
                'source_magnitude': mag,
                'station_code': station_code,
                'trace_P_arrival_sample': arrival_sample,
                'trace_S_arrival_sample': arrival_sample + (5 * 100),
                'start_time': str(t_start)
            })

            saved_count += 1
            downloaded_event = True

df = pd.DataFrame(metadata_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Файл данных: {OUTPUT_H5}")
print(f"Файл метаданных: {OUTPUT_CSV}")
