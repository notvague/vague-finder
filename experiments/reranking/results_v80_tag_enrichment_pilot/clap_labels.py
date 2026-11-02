"""소리 라벨 고정표. (라벨, CLAP 문장, 태그 표기, dev 단서 표기). 결과를 본 뒤 고치지 않는다."""
LABELS = (
    ("피아노", "a song with prominent piano", ("피아노",), ("피아노", "건반")),
    ("통기타", "a song with prominent acoustic guitar", ("통기타", "어쿠스틱기타", "어쿠스틱"), ("통기타", "어쿠스틱", "어쿠스틱기타")),
    ("일렉기타", "a song with prominent electric guitar", ("일렉기타", "일렉트릭기타", "기타리프"), ("일렉기타", "기타리프", "일렉트릭기타")),
    ("스트링", "a song with a string section of violins and cellos", ("스트링", "현악", "바이올린", "첼로"), ("스트링", "현악기", "현악", "바이올린", "첼로")),
    ("오케스트라", "a song with a full orchestra", ("오케스트라", "관현악"), ("오케스트라", "관현악")),
    ("신스", "a song with synthesizers and electronic sounds", ("신스", "신디사이저", "전자음", "일렉트로닉"), ("신스", "신디사이저", "전자음", "일렉트로닉")),
    ("드럼", "a song with prominent drums", ("드럼",), ("드럼",)),
    ("브라스", "a song with a brass section of trumpets and saxophones", ("브라스", "트럼펫", "색소폰", "관악"), ("브라스", "트럼펫", "색소폰")),
    ("고음", "a singer belting very high powerful notes", ("고음", "샤우팅"), ("고음", "삼단고음")),
    ("디스코", "a disco song", ("디스코",), ("디스코",)),
    ("랩", "a person rapping", ("랩", "래핑", "랩핑"), ("랩", "래핑")),
)
TOP_SHARE = 0.15
