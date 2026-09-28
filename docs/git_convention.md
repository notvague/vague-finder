# Git & PR Convention

팀원 간의 원활한 협업과 명확한 히스토리 관리를 위해 아래 규칙을 준수해주세요.
서로의 아이디(ID)만으로는 누가 어떤 작업을 했는지 식별하기 어려우므로, **실명**을 포함하는 것을 원칙으로 합니다.

## 1. Branch Naming (브랜치 이름)

작업 종류와 간단한 설명을 소문자 영어로 작성합니다.
여러개면 대표적인거 하나만 적어도 됩니다.

- **Format**: `type/description`
- **Types**:
  - `feat/`: 새로운 기능 추가
  - `fix/`: 버그 수정
  - `docs/`: 문서 수정
  - `refactor/`: 코드 리팩토링
  - `chore/`: 빌드, 패키지 매니저 등 설정 파일 변경
- **Example**:
  - `feat/youtube-crawler`
  - `fix/encoding-error`

## 2. PR (Pull Request) Title

PR 제목에는 **작업 종류**와 **내용**, 그리고 **작업자 이름**을 명시합니다.

- **Format**: `[Type] Title - Name`
- **Example**:
  - `[Feat] 유튜브 크롤러 댓글 필터링 구현 - 이연우`
  - `[Fix] 임베딩 모델 로드 에러 수정 - 최정현`
  - `[Docs] API 명세서 업데이트 - 황찬혁`
