---
imageNameKey: 2025-03-20-14-31
tags:
  - LGES_PROJECT
---
- 더블클릭 이벤트
	- ![[2025-03-20-14-31_measure result 측정치 viewer-5.png]]
	![[2025-03-20-14-31_measure result 측정치 viewer-7.png]]
- hDefect
	- inspectresult에서 나온 결함을 저장하고 있음
	- void CLGESInspectionDlg::UpdateDefectList(const E_TypeProperty &e, HObject &defectResult) 에서 Update 
	- ![[2025-03-20-14-31_measure result 측정치 viewer.png]]

UpdateDefectList 사용
- ![[2025-03-20-14-31_measure result 측정치 viewer-1.png]]
- ![[2025-03-20-14-31_measure result 측정치 viewer-2.png]]
- ![[2025-03-20-14-31_measure result 측정치 viewer-3.png]]
- ![[2025-03-20-14-31_measure result 측정치 viewer-4.png]]


검사 함수
![[2025-03-20-14-31_measure result 측정치 viewer-6.png]]


HInspection.h에 있는 HTupleVector의 measureResults에 측정치들이 저장됨
idx는 measureProcessor의 번호로 결과값을 가져올 수 있음


- 더블클릭 이벤트에서 클릭한 점을 포함하는 region을 hdefects 중에 하나를 SelectRegionPoint 함수를 통해 가져옴
- hdefects는 defectprocess에서 생성된 결함 영역들을 포함하고 있음,  이 hdefect는 BuildResult에서 생성되고 HInspectResult Class에 HObject defectResult로 선언되어 있음
- 먼저 Inpection에서 defectprocess에서 defProcRegion에 해당 프로세스에서 나온 결함을 inspResult.defProcRegions에 가져오고 SelectDefects에서 inspResult.selectedDefect에 선택된 결함이 저장된다.

