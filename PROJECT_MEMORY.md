# Project Memory

Last Updated: 2026-06-27 20:14 Asia/Shanghai

## 妞ゅ湱娲伴惄顔界垼

- 鏋勫缓涓€鏉′繚韬唤鐨勪汉鍍忓鍙戣縼绉讳骇鍝侀摼璺紝浼樺厛淇濊瘉鈥滆繕鏄湰浜衡€濓紝鍐嶉€愭澧炲己鍙戝瀷鍜屽瀹瑰鍙傝€冨浘鐨勮创鍚堝害銆?- 褰撳墠闃舵鐩爣鏄皢 MVP 浠庡崟娆″急鍙傝€?inpaint 鍒囨崲鍒颁袱闃舵灞€閮ㄧ紪杈戯細鍏堝彂鍨嬨€佸悗濡嗗銆佽劯鍜岄厤楗板叏绋嬮攣姝汇€?
## 瑜版挸澧犻弸鑸电€?
- 鍚庣涓?Python / FastAPI 椋庢牸鐨勪换鍔＄紪鎺掓湇鍔★紝鍖呭惈 preprocess銆乺eference parser銆乬enerator銆乸ostprocess銆乻coring銆乤rtifact persistence銆?- 褰撳墠涓荤敓鎴?Provider 涓?Ark HTTP inpaint锛岃姹傚眰浼氬彂閫?source + reference锛屽苟闄勫甫缁撴瀯鍖栧鍙戠壒寰佸拰闃舵鍖栨帶鍒跺瓧娈点€?- 2026-06-27 璧凤紝`full_transfer` 鍦ㄤ繚鐣欓厤楗板満鏅笅榛樿璧?`two_stage_local_edit`锛屼笉鏄崟娆?`local_inpaint`銆?
## 瑜版挸澧?Provider

- `GENERATION_PROVIDER=ark_http`
- `ARK_INPAINT_MODEL=jimeng_image2image_dream_inpaint`
- `ARK_MODEL=jimeng_t2i_v40` 瀛樺湪浜庨厤缃腑锛屼絾褰撳墠瀹為檯鐢熸垚涓婚摼璺娇鐢ㄧ殑鏄?`ARK_INPAINT_MODEL`銆?
## 瀹告彃鐣幋鎰杽妤?

### [2026-06-27 15:41 Asia/Shanghai] Two-stage local edit wiring smoke test
- 閺冨爼妫? 2026-06-27 15:41 Asia/Shanghai
- 鐎圭偤鐛欓崥宥囆? Two-stage local edit wiring smoke test
- 鐎圭偤鐛欓柊宥囩枂: `scripts/run_random_generation.py --source "D:\姘存湪骞村崕\娴嬭瘯鍥綷origin2.jpg" --reference "D:\姘存湪骞村崕\娴嬭瘯鍥綷reference7.jpg" --candidate-count 1`
- 鐎圭偤鐛欑紒鎾寸亯: 浠诲姟 `job_ebf153e5dbe642229d7fdc5fff01d6f8` 鎴愬姛锛宍selected_pipeline=two_stage_local_edit`锛岀敓鎴愪簡 hair stage 鏈湴杈撳嚭鍜?makeup stage 鏈€缁堣緭鍑猴紱鏈€缁?metadata 涓褰曚簡闃舵 runs 鍜?stage prompts銆?- 鐎圭偤鐛欑紒鎾诡啈: 涓ら樁娈靛眬閮ㄧ紪杈戠殑缂栨帓銆侀樁娈?source 涓叉帴銆佹湰鍦颁腑闂寸粨鏋滀繚瀛樺拰鏈€缁堝厓鏁版嵁钀界洏鍧囧凡鎵撻€氥€?
### [2026-06-27 17:05 Asia/Shanghai] Source mask generation upgrade
- 閺冨爼妫? 2026-06-27 17:05 Asia/Shanghai
- 鐎圭偤鐛欓崥宥囆? Source mask generation upgrade
- 鐎圭偤鐛欓柊宥囩枂: 瀵?`origin2.jpg` 杩愯鏂扮殑鍘熷浘棰勫鐞嗭紝鐢熸垚 `source_hair_edit_mask / source_makeup_edit_mask / source_face_lock_mask / source_accessory_mask` 骞朵汉宸ユ鏌ャ€?- 鐎圭偤鐛欑紒鎾寸亯: 鍘熷浘 mask 宸蹭粠 mock 璧勪骇鍒囨崲涓虹湡瀹炶惤鐩樻枃浠讹紱`makeup_mask` 鍜?`face_lock_mask` 宸叉槑鏄炬敹鏁涘埌闈㈤儴灞€閮ㄥ尯鍩燂紱`hair_mask` 澧炲姞浜嗕笅閲囨牱绾︽潫寮?GrabCut 鍒嗘敮鍚庯紝鐩告瘮涓婁竴鐗堝噺灏戜簡閮ㄥ垎鑴搁儴渚佃殌锛屼絾浠嶆湁椤堕儴鑳屾櫙璇叆銆?- 鐎圭偤鐛欑紒鎾诡啈: 鍘熷浘灞€閮ㄧ紪杈戞帶鍒跺凡缁忚繘鍏モ€滅湡 mask鈥濋樁娈碉紝鍙戝瀷鍖轰粛鏄綋鍓嶆渶澶х煭鏉匡紝鍚庣画搴旂户缁彁鍗?hair segmentation 鎴栧紑濮嬬敤鐜版湁 mask 鍋氫竴娆＄湡瀹炵敓鎴愬姣旈獙璇併€?
## 瀹稿弶甯撻梽銈夋６妫?

- 宸叉帓闄も€滃綋鍓?`full_transfer` 浠嶇劧鍙細璧板崟娆?local inpaint鈥濈殑鎯呭喌锛?026-06-27 鐨勭湡瀹炶繍琛岀粨鏋滄樉绀?`selected_pipeline=two_stage_local_edit`銆?
## 瑜版挸澧犻梻顕€顣?
- 棰勫鐞嗗眰鐨?`editable_hair_mask`銆乣editable_makeup_mask`銆乣face_lock_mask` 鐩墠浠嶄负鍗犱綅璧勪骇 URI锛屽皻鏈帴鍏ョ湡瀹?segmentation 杈撳嚭銆?- 璐ㄩ噺璇勫垎浠嶇劧鏄惎鍙戝紡鍒嗘暟锛屼笉鑳藉崟鐙綔涓衡€滅粨鏋滆瑙夋纭€濈殑鍒ゆ柇渚濇嵁銆?- 涓ら樁娈靛凡鎵撻€氾紝浣嗙粨鏋滆川閲忔槸鍚﹁冻澶熶骇鍝佸寲浠嶉渶缁х画鍋氱湡瀹?mask銆侀樁娈典笓鐢ㄥ悗澶勭悊鍜屾洿鍙俊鐨勮川妫€銆?- `source_hair_edit_mask` 宸叉湁鐪熷疄杈撳嚭锛屼絾椤堕儴鑳屾櫙璇叆浠嶆槑鏄撅紝鍙戝瀷闃舵鎺у埗鍔涜繕涓嶅浜у搧绾с€?
## 瑜版挸澧犻幎鈧張顖濈熅缁?

- 褰撳墠纭璺嚎涓猴細`full_transfer -> two_stage_local_edit`
- stage 1: `hair_only inpaint`
- stage 2: `makeup_only inpaint`
- 涓ら樁娈甸兘鍙戦€?source + reference锛屽苟鍦?prompt / req_json 涓樉寮忓０鏄庨樁娈电洰鏍囥€佹椿鍔ㄧ紪杈戞帺鐮併€乫ace lock銆乸arent candidate銆?
## 娑撳绔村銉吀閸?

- 灏嗛澶勭悊杈撳嚭浠庡崰浣?mask 鍗囩骇涓虹湡瀹?segmentation 妯″瀷缁撴灉锛屼紭鍏堟帴鍏?hair / makeup / face lock 鍙紪杈戝尯鍩熴€?- 涓?hair stage 鍜?makeup stage 鍒嗗埆澧炲姞鏇村己鐨勮川妫€閫昏緫锛岄伩鍏嶅綋鍓嶅惎鍙戝紡楂樺垎璇垽銆?- 璇勪及闃舵缁撴灉鍥撅紝閽堝 identity drift銆佸彂鍨嬪亸宸€佸瀹瑰亸宸ˉ鍏呮洿涓ユ牸鐨?prompt 鍜屽悗澶勭悊鎺у埗銆?- 鐢ㄦ柊鐨?source masks 璺戜竴娆′袱闃舵鐪熷疄鐢熸垚锛岄獙璇佺粨鏋滃浘鏄惁杈冧箣鍓嶅噺灏戣劯閮ㄦ敼鍐欎笌鍙戝瀷婕傜Щ銆?
## 閺堚偓鏉╂垳鎱ㄩ弨瑙勬瀮娴?

### [2026-06-27 15:50 Asia/Shanghai] File updates
- 閺傚洣娆㈤崥? app/models/pipeline.py
  娣囶喗鏁奸崢鐔锋礈: 鏂板 `editable_makeup_mask` 涓?`face_lock_mask`锛屼负涓ら樁娈靛眬閮ㄧ紪杈戞彁渚涙暟鎹帴鍙ｃ€?- 閺傚洣娆㈤崥? app/services/preprocess.py
  娣囶喗鏁奸崢鐔锋礈: 杈撳嚭涓ら樁娈垫墍闇€鐨勫崰浣嶆帺鐮佷笌 `two_stage_local_edit_ready` 璐ㄩ噺鏍囪銆?- 閺傚洣娆㈤崥? app/services/generator.py
  娣囶喗鏁奸崢鐔锋礈: 灏?`full_transfer + preserve_accessories` 璺敱鍒?`two_stage_local_edit`锛屽苟璁╂湰鍦?inpaint worker 鎺ユ敹闃舵涓婁笅鏂囥€?- 閺傚洣娆㈤崥? app/services/model_clients.py
  娣囶喗鏁奸崢鐔锋礈: 閲嶅啓璇锋眰鏋勯€狅紝鏀寔 hair / makeup 闃舵 prompt銆侀樁娈?source 涓叉帴銆乤ctive edit mask銆乫ace lock 鍏冩暟鎹€?- 閺傚洣娆㈤崥? app/services/orchestrator.py
  娣囶喗鏁奸崢鐔锋礈: 閲嶅啓涓轰袱闃舵鎵ц鍣紝淇濆瓨 hair stage 涓棿缁撴灉骞跺皢鍏舵帴鍏?makeup stage銆?- 閺傚洣娆㈤崥? app/services/scoring.py
  娣囶喗鏁奸崢鐔锋礈: 涓?`two_stage_local_edit` 鍗曠嫭璋冩暣鍚彂寮?identity / transfer / artifact 鏉冮噸銆?- 閺傚洣娆㈤崥? scripts/run_random_generation.py
  娣囶喗鏁奸崢鐔锋礈: 娓呯悊鏃х紪鐮侀棶棰橈紝骞惰緭鍑洪樁娈靛厓鏁版嵁鐢ㄤ簬鏈湴楠岃瘉銆?### [2026-06-27 17:05 Asia/Shanghai] File updates
- 閺傚洣娆㈤崥? app/services/preprocess.py
  娣囶喗鏁奸崢鐔锋礈: 灏嗗師鍥鹃澶勭悊鍗囩骇涓虹湡瀹?mask 钀界洏锛屽苟鏂板鏇村己鐨?source hair mask 閫昏緫銆?- 閺傚洣娆㈤崥? app/utils/images.py
  娣囶喗鏁奸崢鐔锋礈: 澧炲己鏈湴鏂囦欢璺緞璇诲彇锛岄伩鍏嶄腑鏂囪矾寰勮鍒や负 base64銆?
### [2026-06-27 18:05 Asia/Shanghai] Ark success-path alignment audit
- 鏃堕棿: 2026-06-27 18:05 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 鎴愬姛閾捐矾瀵归綈鏍告煡
- 瀹為獙閰嶇疆: 瀵规瘮 `.env.local`銆乣app/services/model_clients.py`銆乣app/services/generator.py` 鐨勫疄闄呰皟鐢ㄥ舰鎬侊紝骞舵牳瀵圭伀灞卞畼鏂?API / 寮€鍙戣€呮枃绔犱腑 `JimengImage2ImageDreamInpaintSubmitTask`銆乣鍗虫ⅵAI 鍥剧敓鍥?3.0 鏅鸿兘鍙傝€僠銆乣鏅鸿兘鍙傝€冨浘鐢熷浘`銆丼eedream 澶氬浘鍙傝€冭兘鍔涙弿杩般€?- 瀹為獙缁撴灉: 褰撳墠浠撳簱鐪熷疄璋冪敤鐨勬槸 `jimeng_image2image_dream_inpaint`锛屽浘鍍忚緭鍏ヤ负 `source + active_edit_mask`锛沗reference_image` 浠呰繘鍏?`prompt` / `req_json`锛宍global_reference` 浠嶄负 mock锛涘畼鏂瑰叕寮€鏉愭枡鍚屾椂瀛樺湪鈥滃眬閮ㄩ噸缁?inpaint鈥濅笌鈥滄櫤鑳藉弬鑰冨浘鐢熷浘 / 鍥剧敓鍥?3.0 鏅鸿兘鍙傝€?/ 澶氬浘铻嶅悎鈥濅袱绫讳笉鍚岃兘鍔涜〃杩般€?- 瀹為獙缁撹: 褰撳墠浠ｇ爜瀹為檯璋冪敤鐨勮兘鍔涘舰鎬佷笌鈥滀繚韬唤 + 寮哄弬鑰冨浘濡嗗彂杩佺Щ涓婚摼璺€濅笉涓€鑷达紱褰撳墠鏈€鍚堢悊缁撹涓?C锛氱己鐨勬槸鍙︿竴鏉℃ā鍨嬫垨鎺ュ彛锛岀幇鏈?`two_stage_local_edit + jimeng_image2image_dream_inpaint` 搴旇涓哄厹搴曢摼璺€?
## 2026-06-27 Ark 瀵归綈鏍告煡琛ュ厖

- 褰撳墠宸插舰鎴愮嫭绔嬫牳鏌ユ枃妗?`Ark鎴愬姛閾捐矾瀵归綈鏍告煡.md`锛岀敤浜庢柊瀵硅瘽鐩存帴鎺ユ墜銆?- 褰撳墠涓嶅缓璁户缁妸 `jimeng_image2image_dream_inpaint` 褰撲富閾捐矾璋冩晥鏋滐紝涓嬩竴姝ュ簲鍏堢‘璁ょ湡姝ｅ彲鎺ョ殑 Ark 寮哄弬鑰冨浘濡嗗彂杩佺Щ鑳藉姏銆?
### [2026-06-27 18:05 Asia/Shanghai] File updates
- 鏂囦欢鍚? Ark鎴愬姛閾捐矾瀵归綈鏍告煡.md
  淇敼鍘熷洜: 钀界洏 Ark 鎴愬姛閾捐矾瀵归綈鏍告煡锛屾槑纭綋鍓?inpaint 閾捐矾涓庣洰鏍囦富閾捐矾鐨勫樊寮傦紝骞剁粰鍑?A/B/C 缁撹涓庝笅涓€姝ュ缓璁€?- 鏂囦欢鍚? PROJECT_MEMORY.md
  淇敼鍘熷洜: 杩藉姞鏈 Ark 鑳藉姏瀵归綈鏍告煡缁撹锛屼緵鍚庣画瀵硅瘽閬垮厤閲嶅璇曢敊銆?
### [2026-06-27 18:18 Asia/Shanghai] Ark path wording correction
- 鏃堕棿: 2026-06-27 18:18 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 涓婚摼璺彛寰勪慨姝?- 瀹為獙閰嶇疆: 鏍规嵁鐢ㄦ埛琛ュ厖鐨勬垚鍔熼摼璺鏄庝笌椤甸潰渚т骇鍝佹灦鏋勬弿杩帮紝閲嶅啓 Ark 瀵归綈鏍告煡鏂囨。鐨勭粨璁哄彛寰勩€?- 瀹為獙缁撴灉: 褰撳墠缁熶竴鍙ｅ緞淇涓衡€淎rk 浠嶆槸涓婚摼璺钩鍙帮紝浣嗗綋鍓嶄粨搴撳彧鍛戒腑浜嗗叕寮€ `jimeng_image2image_dream_inpaint` 瀛愯兘鍔涳紝娌℃湁澶嶇幇鎴愬姛鏃堕偅濂?Ark 瀹屾暣娴佹按绾垮師鐢熷簳灞?+ 宸ョ▼鎺у埗闂幆鈥濄€?- 瀹為獙缁撹: 鍚庣画涓嶅簲鍐嶆妸闂琛ㄨ堪鎴愨€淎rk 涓嶆槸涓婚摼璺€濓紝鑰屽簲琛ㄨ堪涓衡€滃綋鍓嶄粨搴撴病鏈夊榻愬埌 Ark 瀹屾暣涓婚摼璺紝鍙疄鐜颁簡 Ark 鍏滃簳 inpaint 瀛愰摼璺€濄€?
### [2026-06-27 18:18 Asia/Shanghai] File updates
- 鏂囦欢鍚? Ark鎴愬姛閾捐矾瀵归綈鏍告煡.md
  淇敼鍘熷洜: 鎸夋渶鏂版垚鍔熼摼璺鏄庨噸鍐欐牳鏌ユ枃妗ｏ紝鏄庣‘鍖哄垎 Ark 瀹屾暣鍘熺敓娴佹按绾夸笌褰撳墠浠撳簱鍛戒腑鐨勫叕寮€ inpaint 瀛愯兘鍔涖€?
### [2026-06-27 18:24 Asia/Shanghai] Ark mainline implementation task draft
- 鏃堕棿: 2026-06-27 18:24 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 瀹屾暣涓婚摼璺爺鍙戜换鍔″崟鏁寸悊
- 瀹為獙閰嶇疆: 鍩轰簬 Ark 鎴愬姛閾捐矾瀵归綈鏍告煡缁撴灉锛屽皢鍚庣画宸ヤ綔鏁寸悊涓烘爣鍑嗙爺鍙戜换鍔″崟锛屾槑纭洰鏍囥€侀潪鐩爣銆佸緟鍔炴媶瑙ｃ€佹墽琛岄『搴忓拰楠屾敹鏍囧噯銆?- 瀹為獙缁撴灉: 宸插舰鎴?`Ark瀹屾暣涓婚摼璺爺鍙戜换鍔″崟.md`锛屽皢鍚庣画宸ヤ綔缁熶竴涓衡€滃厛纭 Ark 瀹屾暣涓婚摼璺兘鍔涳紝鍐嶅喅瀹氫粨搴撳浣曡ˉ涓婚摼璺紝涓嶅啀鐩茶皟褰撳墠 inpaint鈥濄€?- 瀹為獙缁撹: 鍚庣画寮€鍙戝彲浠ョ洿鎺ユ寜浠诲姟鍗曟帹杩?P0/P1锛岃€屼笉闇€瑕佸啀娆￠噸澶嶆緞娓呬富閾捐矾涓庡厹搴曢摼璺殑鍖哄埆銆?
### [2026-06-27 18:24 Asia/Shanghai] File updates
- 鏂囦欢鍚? Ark瀹屾暣涓婚摼璺爺鍙戜换鍔″崟.md
  淇敼鍘熷洜: 灏?Ark 涓婚摼璺榻愬伐浣滄暣鐞嗘垚鏍囧噯鐮斿彂浠诲姟鍗曪紝鏂逛究鍚庣画鐩存帴鎸夌洰鏍囥€佽竟鐣屻€佸緟鍔炲拰楠屾敹鎺ㄨ繘銆?
### [2026-06-27 18:38 Asia/Shanghai] Ark mainline/fallback code split
- 鏃堕棿: 2026-06-27 18:38 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 涓婚摼璺笌 inpaint 鍏滃簳閾捐矾浠ｇ爜鍒嗗眰
- 瀹為獙閰嶇疆: 淇敼 `app/config.py`銆乣app/models/pipeline.py`銆乣app/services/generator.py`銆乣app/services/orchestrator.py`锛屽皢 `full_transfer` 鐨勯粯璁ゅ喅绛栦粠鐩存帴杩涘叆涓ら樁娈?inpaint锛屾敼涓衡€滃厛灏濊瘯 Ark 涓婚摼璺紝鍐嶅湪涓嶅彲鐢ㄦ椂鍥為€€鍒?`two_stage_local_edit`鈥濓紝骞惰褰?pipeline decision / attempts銆?- 瀹為獙缁撴灉: 浠撳簱宸插叿澶囨樉寮忕殑涓婚摼璺喅绛栧璞°€佷富閾捐矾鍗犱綅 worker銆佸厹搴曞洖閫€閫昏緫锛屼互鍙婅惤鐩樺埌 job metadata 鐨?pipeline attempts锛涜交閲忕儫娴嬫樉绀?`full_transfer` 浼氬厛閫夋嫨 `ark_complete_mainline`锛屽湪涓婚摼璺湭鎺ュ叆鏃惰嚜鍔ㄥ洖閫€鍒?`two_stage_local_edit`銆?- 瀹為獙缁撹: 褰撳墠浠撳簱鐨勪唬鐮佸績鏅烘ā鍨嬪凡浠庘€滈粯璁ゆ妸 inpaint 褰撲富閾捐矾鈥濅慨姝ｄ负鈥淎rk 涓婚摼璺紭鍏堬紝inpaint 浠呬綔鍏滃簳鈥濓紝鍚庣画鍙互鍦ㄦ鍩虹涓婄户缁帴鐪熷疄涓婚摼璺粦瀹氥€?
### [2026-06-27 18:38 Asia/Shanghai] File updates
- 鏂囦欢鍚? app/config.py
  淇敼鍘熷洜: 鏂板 Ark 涓婚摼璺紑鍏充笌涓婚摼璺厤缃垽鏂紝鏀寔浠ｇ爜灞傚尯鍒嗕富閾捐矾涓庡厹搴曢摼璺€?- 鏂囦欢鍚? app/models/pipeline.py
  淇敼鍘熷洜: 鏂板 pipeline decision / attempt 鏁版嵁缁撴瀯锛岀敤浜庤褰曚富閾捐矾鍐崇瓥鍜屽洖閫€杞ㄨ抗銆?- 鏂囦欢鍚? app/services/generator.py
  淇敼鍘熷洜: 灏嗙敓鎴愯矾鐢辨敼涓衡€淎rk 涓婚摼璺紭鍏堛€佷袱闃舵 inpaint 鍏滃簳鈥濓紝骞舵柊澧炰富閾捐矾鍗犱綅 worker 涓?inpaint attempt 璁板綍銆?- 鏂囦欢鍚? app/services/orchestrator.py
  淇敼鍘熷洜: 灏嗙紪鎺掑眰鏀逛负鏀寔涓婚摼璺皾璇曘€佽嚜鍔ㄥ洖閫€鍜?pipeline metadata 钀界洏銆?
### [2026-06-27 18:52 Asia/Shanghai] Ark mainline API binding implementation
- 鏃堕棿: 2026-06-27 18:52 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 瀹屾暣涓婚摼璺湡瀹炶姹傜粦瀹?- 瀹為獙閰嶇疆: 鍦?`app/services/model_clients.py` 涓负 `ArkVisualClient` 鏂板 mainline 璋冪敤璺緞锛屾寜 `source + reference` 鍙屽浘杈撳叆鏋勯€犱富閾捐矾璇锋眰锛涘綋涓绘ā鍨嬩负 `jimeng_t2i_v40` 涓旀湭鏄惧紡瑕嗙洊 action 鏃讹紝鑷姩瑙ｆ瀽涓?`JimengT2IV40SubmitTask / JimengT2IV40GetResult`锛涘苟鍦?`app/services/generator.py` 灏?`ArkMainlineWorker` 鏀逛负鐪熷疄璋冪敤璇ヨ矾寰勩€?- 瀹為獙缁撴灉: 浠撳簱宸插叿澶囩湡瀹炰富閾捐矾璇锋眰缁戝畾浠ｇ爜锛屼笉鍐嶅彧鏄崰浣?worker锛涚紪璇戞鏌ラ€氳繃锛岃交閲忕儫娴嬭兘姝ｇ‘璁板綍涓婚摼璺?action 瑙ｆ瀽缁撴灉骞跺湪寮傚父鏃跺洖閫€鍒?`two_stage_local_edit`銆?- 瀹為獙缁撹: 褰撳墠浠撳簱宸茬粡浠庘€滀富閾捐矾鍗犱綅鈥濇帹杩涘埌鈥滀富閾捐矾鐪熷疄缁戝畾浠ｇ爜宸叉帴鍏モ€濓紝浣嗗皻鏈畬鎴愬熀浜庣湡瀹炰粯璐?API 鐨勭鍒扮鏁堟灉楠岃瘉锛屽洜姝や笉鑳芥妸涓婚摼璺涓哄凡楠屾敹銆?
### [2026-06-27 18:52 Asia/Shanghai] File updates
- 鏂囦欢鍚? app/services/model_clients.py
  淇敼鍘熷洜: 鏂板 Ark mainline 鍙屽浘杈撳叆璇锋眰鏋勯€犮€佷富閾捐矾 prompt銆乤ction 鑷姩瑙ｆ瀽涓庣粨鏋滆В鏋愰€昏緫銆?- 鏂囦欢鍚? app/services/generator.py
  淇敼鍘熷洜: 灏?Ark 涓婚摼璺?worker 浠庡崰浣嶅疄鐜板垏鎹负鐪熷疄 API 缁戝畾璋冪敤锛屽苟鍦?attempt metadata 涓褰曚富閾捐矾 action銆?
### [2026-06-27 19:20 Asia/Shanghai] Ark mainline real smoke test
- 鏃堕棿: 2026-06-27 19:20 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 瀹屾暣涓婚摼璺湡瀹炵儫娴?- 瀹為獙閰嶇疆: 浣跨敤 `scripts/run_random_generation.py --source "D:\姘存湪骞村崕\娴嬭瘯鍥綷origin2.jpg" --reference "D:\姘存湪骞村崕\娴嬭瘯鍥綷reference7.jpg" --candidate-count 1`锛屽湪褰撳墠鐪熷疄 `.env.local` 涓嬭繍琛岋紱`full_transfer` 鍏堝皾璇?`ark_complete_mainline`锛屼富妯″瀷涓?`jimeng_t2i_v40`锛岃嚜鍔ㄨВ鏋?action 涓?`JimengT2IV40SubmitTask / JimengT2IV40GetResult`銆?- 瀹為獙缁撴灉: 浠诲姟 `job_7a581ed43e7341a78b0eca121046c8e0` 鎴愬姛瀹屾垚锛屼絾 `pipeline_attempts` 鏄剧ず涓婚摼璺皾璇曡 Ark 渚ф嫆缁濓紝閿欒涓?`no such api`锛涚郴缁熼殢鍚庤嚜鍔ㄥ洖閫€鍒?`two_stage_local_edit`锛岀敱 `jimeng_image2image_dream_inpaint` 鐨?hair stage + makeup stage 鎴愬姛鍑哄浘骞惰惤鐩樸€?- 瀹為獙缁撹: 褰撳墠涓婚摼璺唬鐮佺粦瀹氬凡鐪熷疄鍙戣捣璇锋眰锛屼絾鈥渀jimeng_t2i_v40` + `JimengT2IV40SubmitTask/GetResult`鈥濊繖缁?action 鍦ㄥ綋鍓嶈处鍙?/ 鏈嶅姟绔帴鍙ｄ笂涓嶅彲鐢紱涓嬩竴姝ュ簲浼樺厛纭鐪熷疄鍙敤鐨?Ark mainline action / req_key 瀵癸紝鑰屼笉鏄户缁皟褰撳墠鍙屽浘璇锋眰浣撶粏鑺傘€?
### [2026-06-27 19:20 Asia/Shanghai] File updates
- 鏂囦欢鍚? PROJECT_MEMORY.md
  淇敼鍘熷洜: 璁板綍 Ark 涓婚摼璺涓€娆＄湡瀹炵儫娴嬬粨鏋滐紝鏄庣‘褰撳墠澶辫触鐐瑰湪鏈嶅姟绔?`no such api`锛屽苟纭浠诲姟鏈€缁堟槸閫氳繃鍏滃簳 inpaint 閾捐矾瀹屾垚銆?### [2026-06-27 20:14 Asia/Shanghai] Ark mainline no-such-api fix verification
- 鏃堕棿: 2026-06-27 20:14 Asia/Shanghai
- 瀹為獙鍚嶇О: Ark 涓婚摼璺?no such api 淇楠岃瘉
- 瀹為獙閰嶇疆: 浣跨敤 `py -3 scripts/run_random_generation.py --source "D:\姘存湪骞村崕\娴嬭瘯鍥綷origin2.jpg" --reference "D:\姘存湪骞村崕\娴嬭瘯鍥綷reference7.jpg" --candidate-count 1`锛屽湪 `.env.local` 涓嬪璺戠湡瀹炰富閾捐矾锛屽綋鍓?`ARK_ACTION/ARK_GET_ACTION` 涓?`CVSync2AsyncSubmitTask/CVSync2AsyncGetResult`锛孉rk mainline req_key 涓?`jimeng_t2i_v40`銆?- 瀹為獙缁撴灉: 浠诲姟 `job_e8ed721b27e74b7484c6e299926e407d` 鎴愬姛瀹屾垚锛宍selected_pipeline=ark_complete_mainline`锛宲ipeline attempts 浠呮湁涓€鏉?`ark_complete_mainline -> succeeded`锛宻ubmit/get action 鏄庣‘涓?`CVSync2AsyncSubmitTask/CVSync2AsyncGetResult`锛屾病鏈夊洖閫€鍒?`two_stage_local_edit`銆?- 瀹為獙缁撹: `no such api` 鐨勪富閾捐矾澶辫触鐐瑰凡琚慨澶嶏紱褰撳墠涓婚摼璺凡缁忚兘鐪熷疄璺戣捣鏉ワ紝鍚庣画鐨勪富瑕侀棶棰樹笉鍐嶆槸鎺ュ彛鍙敤鎬э紝鑰屾槸鐢熸垚缁撴灉璐ㄩ噺涓庘€滀繚鏈汉 + 瀹屾暣杩佺Щ濡嗗彂鈥濈殑浜у搧鐩爣浠嶆湁宸窛銆?### [2026-06-27 20:14 Asia/Shanghai] File updates
- 鏂囦欢鍚? pyproject.toml
  淇敼鍘熷洜: 琛ュ叏 Ark 涓婚摼璺疄闄呰繍琛屾墍闇€鐨勪緷璧栧０鏄庯紙`pillow`銆乣requests`銆乣pytz`銆乣pycryptodome`銆乣volcengine`锛夛紝閬垮厤鍥犵幆澧冪己灏?SDK 鎴栧簳灞傚簱鑰屽湪瀵煎叆闃舵澶辫触銆?- 鏂囦欢鍚? PROJECT_MEMORY.md
  淇敼鍘熷洜: 琛ュ厖璁板綍 `no such api` 宸蹭慨澶嶅苟瀹屾垚鐪熷疄涓婚摼璺儫娴嬬殑浜嬪疄缁撴灉锛屼緵鍚庣画鐩存帴杞叆鈥滀富閾捐矾鏁堟灉璋冧紭鈥濄€?

## 已完成实验

### [2026-06-27 19:16 Asia/Shanghai] P0 control bundle and hybrid routing refactor
- 时间: 2026-06-27 19:16 Asia/Shanghai
- 实验名称: P0 control bundle and hybrid routing refactor
- 实验配置: 重构 `app/models/pipeline.py`、`app/services/generator.py`、`app/services/model_clients.py`、`app/services/orchestrator.py`，新增 `GenerationControlBundle`、Ark capability probe、`ark_native_control_mainline / ark_hybrid_mainline` 路由和 control bundle 落盘。
- 实验结果: 编译检查通过；路由干跑结果显示在当前默认配置下，`full_transfer` 会选择 `ark_hybrid_mainline`，fallback 为 `two_stage_local_edit`；control bundle 已可通过 SQLite 回读。
- 实验结论: 在未确认 Ark 原生可执行 mask / identity / control image 支持前，仓库默认不再把 Ark 视为“已具备原生双分支控制”的主链路，而是明确进入 hybrid mainline 模式。

## 当前问题

- 2026-06-27 19:16 Asia/Shanghai: P0 只完成了控制协议和编排模式重构，尚未接入真实 ArcFace/AdaFace 身份编码、真实身份质检和真实后处理回填。
- 2026-06-27 19:16 Asia/Shanghai: `load_image_bytes` 在通过 PowerShell 直接传入包含中文的路径字符串做脚本内联验证时存在路径识别问题，本次未修改运行时文件读写逻辑。

## 当前技术路线

- 2026-06-27 19:16 Asia/Shanghai: 当前确认路线调整为“双线并行”。
- 工程闭环线: 先补 `GenerationControlBundle`、质量门、真实身份编码/质检、后处理回填。
- Ark 能力对齐线: 通过 capability probe 和后续 P5 核查来区分 `ark_native_control_mainline` 与 `ark_hybrid_mainline`，在证据不足前默认走 hybrid。

## 下一步计划

- P1: 在 `app/services/preprocess.py` 接入真实人脸检测与 ArcFace/AdaFace 身份编码。
- P1: 在 `app/services/scoring.py` 用真实人脸相似度替换当前启发式 identity score。
- P3a: 在 `app/services/postprocess.py` 先落地 `accessory_mask` 与核心脸区的最小回填止血逻辑。

## 最近修改文件

### [2026-06-27 19:16 Asia/Shanghai] File updates
- 文件名: app/models/pipeline.py
  修改原因: 新增 `IdentityEmbeddingAsset`、`QualityGate`、`MainlineCapabilityProfile`、`GenerationControlBundle`，把控制协议升级为统一数据结构。
- 文件名: app/config.py
  修改原因: 新增 Ark capability probe 所需配置字段，用于区分是否支持原生可执行控制。
- 文件名: app/services/generator.py
  修改原因: 引入 capability probe，明确 `ark_native_control_mainline / ark_hybrid_mainline / two_stage_local_edit` 三类路由决策。
- 文件名: app/services/model_clients.py
  修改原因: 改为消费 `GenerationControlBundle` 组装 Ark/mainline/inpaint 请求，统一 control bundle 透传。
- 文件名: app/services/orchestrator.py
  修改原因: 在编排层构建并落盘 control bundle，新增 hybrid mainline 编排，并让评分使用 quality gate 阈值。
- 文件名: app/storage/memory_store.py
  修改原因: 新增 control bundle 的 SQLite 持久化与回读。
- 文件名: app/api/routes.py
  修改原因: 在 artifacts 接口中暴露 control bundle，便于后续调试和验收。
- 文件名: app/services/preprocess.py
  修改原因: 将伪身份向量包装为结构化 `IdentityEmbeddingAsset`，为 P1 真实身份编码替换预留接口。
- 文件名: scripts/run_random_generation.py
  修改原因: 输出 control bundle，方便本地验证生成协议和路由结果。

### [2026-06-27 22:30 Asia/Shanghai] P1 real identity gate and P3a stop-loss postprocess
- 时间: 2026-06-27 22:30 Asia/Shanghai
- 实验名称: P1 real identity gate and P3a stop-loss postprocess
- 实验配置: 修改 `app/models/pipeline.py`、`app/services/preprocess.py`、`app/services/postprocess.py`、`app/services/scoring.py`，为 `PreprocessResult` 增加 `source_image_ref`，落地 `accessory hard refill + core face feather blend + masked luminance align`，并让评分层在真实 identity embedding 可用时对候选图执行 ArcFace 余弦相似度校验，对未检测到人脸的候选直接记 0 分。
- 实验结果: `py -3 -m compileall app` 通过；使用 Codex runtime Python 运行 `origin2.jpg -> reference7.jpg` 真实任务时，`preprocess.id_embedding.provider=insightface_arcface_buffalo_l`、`dimension=512`、`quality_flags` 含 `insightface_identity_embedding`；候选图 `postprocess.status=applied`、`accessory_similarity_after=1.0`；同一次任务的真实 `identity_score=0.2763`，因低于 `quality_gate.identity_threshold=0.92` 被判无效，任务最终失败而不是放行；空白候选图验证返回 `scoring_mode=candidate_face_not_detected` 且 `identity_score=0.0`。
- 实验结论: P1 和 P3a 的最小闭环已经接通，系统开始具备“真身份门 + 真回填止血 + 坏结果不放行”的能力；当前主问题已从“评分说假话”转为“分割/主链路效果还不足以通过身份门”。

## 当前问题

- 2026-06-27 22:30 Asia/Shanghai: 默认 `python`/`py -3` 与 Codex runtime Python 不是同一环境，`insightface` 仅在 Codex runtime 中可用；用错解释器做本地冒烟时会退回 `pseudo_preview`。
- 2026-06-27 22:30 Asia/Shanghai: `ark_hybrid_mainline` 当前在真实任务中仍可能产生 `identity_score=0.2763` 的候选，说明仅靠现有 mask 与主链路提示还不足以稳定保住本人脸。

## 下一步计划

- P2: 升级 `editable_hair_mask / editable_makeup_mask / face_lock_mask` 的真实分割质量，优先缩小 `face_lock_mask` 误伤和 `hair mask` 漏编区域。
- P4: 在编排层把当前“筛掉无效候选”升级为“自动重试并记录失败原因”，优先针对 identity failure 和 accessory failure。
- P5: 继续核查 Ark 是否支持更强的原生 control / identity / mask 输入，决定是否能把 `ark_hybrid_mainline` 提升为更强的 native control mainline。

## 最近修改文件

### [2026-06-27 22:30 Asia/Shanghai] File updates
- 文件名: app/models/pipeline.py
  修改原因: 为 `PreprocessResult` 增加 `source_image_ref`，让后处理可以直接取回原图像素做回填和融合。
- 文件名: app/services/preprocess.py
  修改原因: 在预处理结果中透传原图引用，供后处理和后续质量分析使用。
- 文件名: app/services/postprocess.py
  修改原因: 将占位后处理替换为最小可用止血层，落地配饰硬回填、核心脸区 feather blend、亮度对齐与后处理指标回写。
- 文件名: app/services/scoring.py
  修改原因: 接入真实 ArcFace 余弦身份质检，对候选无人脸直接记 0 分，并优先使用后处理回写的真实配饰相似度指标。

### [2026-06-27 23:02 Asia/Shanghai] P2 conservative mask refinement and P4 retry loop skeleton
- 时间: 2026-06-27 23:02 Asia/Shanghai
- 实验名称: P2 conservative mask refinement and P4 retry loop skeleton
- 实验配置: 修改 `app/services/preprocess.py` 与 `app/services/orchestrator.py`，收紧妆区、扩大身份锁定区、补顶部发区恢复，并在编排层新增基于 `quality_gate.max_retry_count` 的自动重试逻辑、失败原因统计、控制强度调参和 identity failure 时的 `two_stage_local_edit` 降级策略。
- 实验结果: `py -3 -m compileall app` 通过；使用 Codex runtime Python 运行 `origin2.jpg -> reference7.jpg` 真实任务时，`quality_flags` 新增 `expanded_hair_edit_mask` 与 `conservative_makeup_mask`；系统共执行 4 轮（`retry_count=3`），第 0 轮为 `ark_hybrid_mainline`，后续 3 轮自动切换为 `two_stage_local_edit`，并逐轮将 `makeup_strength 0.75 -> 0.52`、`hairstyle_strength 0.85 -> 0.68`、`identity_lock_strength 0.95 -> 1.0`；最终候选 `identity_score=0.621`，仍低于 `0.92`，被记录为 `identity_below_threshold` 并拒绝放行。
- 实验结论: P4 的最小可用重试骨架已经落地，系统开始具备“失败原因可见、自动切保守链路、自动收紧参数”的能力；当前主瓶颈仍然是生成结果在真实身份分数上不够高，而不是系统没有重试或没有质量门。

## 当前问题

- 2026-06-27 23:02 Asia/Shanghai: 即便进入 `two_stage_local_edit` 并收紧参数，当前 `origin2.jpg -> reference7.jpg` 任务的最佳真实身份分仍只有 `0.621`，说明现有分割和链路控制还不足以稳定保脸。
- 2026-06-27 23:02 Asia/Shanghai: 当前重试逻辑已可按 identity failure 降级路线，但还没有针对不同失败原因做更细粒度的 mask/Prompt/链路专项重试策略。

## 下一步计划

- P2: 继续提升 `face_lock_mask` 与 `editable_hair_mask` 的几何质量，优先压缩会改动五官的可编辑区域，并补发际线/头顶发区的更稳健恢复。
- P4: 为 `identity_below_threshold`、`accessory_below_threshold`、`transfer_below_threshold` 分别设计差异化重试策略，而不是仅靠统一强度收紧。
- P5: 在 Ark 能力对齐线上继续核查是否存在更强的原生 identity/control/mask 输入面，决定是否值得把当前 hybrid 主链路继续升级。

## 最近修改文件

### [2026-06-27 23:02 Asia/Shanghai] File updates
- 文件名: app/services/preprocess.py
  修改原因: 将源图分割策略调成更保守的妆区与更强的身份锁定区，并补充顶部发区恢复与相关质量标记。
- 文件名: app/services/orchestrator.py
  修改原因: 新增自动重试主循环、失败原因统计、基于 rejection summary 的控制强度调参，以及 identity failure 时切换 `two_stage_local_edit` 的降级逻辑。

### [2026-06-27 23:16 Asia/Shanghai] identity-hard refill and retry prompt tightening
- 时间: 2026-06-27 23:16 Asia/Shanghai
- 实验名称: identity-hard refill and retry prompt tightening
- 实验配置: 修改 `app/services/postprocess.py`、`app/services/model_clients.py`、`app/services/orchestrator.py`，新增更硬的身份核心区回填蒙版、加强 prompt 中对眼型/鼻梁/唇形/下颌线不可改写的约束，并让重试决策在不同失败原因下保留切换空间。
- 实验结果: `py -3 -m compileall app` 通过；使用 Codex runtime Python 再次运行 `origin2.jpg -> reference7.jpg` 真实任务时，后处理新增 `identity_hard_mask_ratio=0.0386` 与 `identity_hard_similarity_after=0.8686` 指标；系统仍执行 4 轮并最终停在 `two_stage_local_edit`；最佳候选真实 `identity_score=0.5678`，低于上一轮的 `0.621`，继续因 `identity_below_threshold` 被拒绝放行。
- 实验结论: 更强的后处理回填和更强的 prompt 约束已经接入，但对当前样本的真实人脸相似度没有带来正向提升，说明仅靠后处理和文案收紧已经接近收益上限，下一步更应该转向更细粒度的 mask 结构或更强的模型控制面。

## 当前问题

- 2026-06-27 23:16 Asia/Shanghai: 当前 `origin2.jpg -> reference7.jpg` 样本在更强后处理与更严 prompt 下，最佳真实身份分从 `0.621` 降到 `0.5678`，说明“更硬回填 + 更强文案约束”不是当前主突破口。

## 下一步计划

- P2: 转向更细粒度的面部禁改区拆分，优先把眼周、鼻梁、唇形、下颌线从现有 face lock 中单独提升为更硬的结构保护区。
- P5: 优先核查 Ark 是否存在真正可执行的 identity/control/mask 能力入口；如果没有，继续在当前链路里做更多 prompt/后处理微调的收益预计有限。

## 最近修改文件

### [2026-06-27 23:16 Asia/Shanghai] File updates
- 文件名: app/services/postprocess.py
  修改原因: 新增 identity hard protect mask 与额外的源图脸区回填指标，进一步加强身份核心区保护。
- 文件名: app/services/model_clients.py
  修改原因: 强化 hair/makeup/mainline prompt 中对眼型、鼻梁、唇形、下颌线等结构不可改写的约束。
- 文件名: app/services/orchestrator.py
  修改原因: 优化重试后的候选衔接与不同失败原因下的链路切换逻辑。

### [2026-06-27 23:31 Asia/Shanghai] P5 Ark public capability alignment conclusion
- 时间: 2026-06-27 23:31 Asia/Shanghai
- 实验名称: P5 Ark public capability alignment conclusion
- 实验配置: 交叉核对火山引擎/即梦官方公开资料与仓库实现，重点核查公开能力面对 `multi-image reference`、`inpaint mask transport`、`identity embedding`、`native executable mask control` 的支持边界；并将结论编码进 `ArkCapabilityProbe`。
- 实验结果: `ArkCapabilityProbe().probe()` 现在稳定返回 `mainline_mode=hybrid`、`evidence_level=official_confirmed`、`confirmed_surfaces=['public_multi_image_reference', 'public_inpaint_mask_transport']`、`missing_surfaces=['native_identity_embedding', 'native_mainline_executable_mask_control']`；`py -3 -m compileall app` 通过。
- 实验结论: 公开可确认的 Ark 能力面只足以支撑“多图参考主链路 + mask inpaint 兜底”这一 hybrid 认知，当前没有官方公开证据支持把仓库声明为“已具备原生可执行 identity embedding / mainline 硬 mask 控制”的 native mainline。

### [2026-06-28 14:34 Asia/Shanghai] Visual identity mode and reference updo-with-bangs alignment
- 时间: 2026-06-28 14:34 Asia/Shanghai
- 实验名称: visual_identity 闭环与参考图盘发刘海对齐
- 实验配置: 修改 `app/models/job.py`、`app/services/orchestrator.py`、`app/services/scoring.py`、`app/services/postprocess.py`、`app/services/reference_parser.py`、`app/services/model_clients.py`、`scripts/run_random_generation.py`，新增 `identity_mode=visual_identity`，为该模式补齐更宽松的质量门/重试/评分策略，保留 Ark hybrid mainline base candidate，修正参考图“盘发+刘海”解析与对应 prompt 约束，并在 visual mode 下优先选择 `accessory_only` 回填与启用视觉身份放行条件。
- 实验结果: `compileall app` 通过；参考图 `reference.png` 解析结果从误判的 `down + side_4_6 + airy_side_bangs` 修正为 `updo + soft_bun + middle + soft_bangs`；使用 Codex runtime Python 运行 `scripts/run_random_generation.py --source origin.jpg --reference reference.png --candidate-count 1 --identity-mode visual_identity` 后，任务 `job_9cd93a3d63e641c08ea817e998b2d4c4` 成功完成，`selected_pipeline=ark_hybrid_mainline`，选中候选为 `hybrid_base_stage` 且 `postprocess_mode=accessory_only`，分数为 `identity_score=0.6333`、`transfer_score=0.84`、`accessory_score=1.0`、`final_score=0.6997`。
- 实验结论: 当前公开 Ark 接口下，仓库已经形成一条可运行的“视觉保本人 + 完整迁移盘发刘海妆容 + 配饰回填”的工程闭环；它仍是 Ark hybrid mainline，而不是服务端原生双分支控制，但对用户目标图样本已经能产出成功结果。

## 当前问题

- 2026-06-28 14:34 Asia/Shanghai: `hybrid_makeup_refine_stage` 在当前账号下仍多次返回 `Access Denied`，因此当前成功样本实际选中的是 Ark mainline base candidate，而不是 refine candidate。
- 2026-06-28 14:34 Asia/Shanghai: 当前 `visual_identity` 的成功标准已经从单纯高 ArcFace 阈值改为“中等身份分 + 高迁移 + 高配饰保留”的组合放行，适合主观视觉目标，但与严格身份锁定模式不同。

## 当前技术路线

- 2026-06-28 14:34 Asia/Shanghai: 当前确认路线增加为“双模式并存”。
- `strict_identity`: 继续维持高 identity gate 与更保守的区域保护。
- `visual_identity`: 优先 Ark hybrid mainline base candidate，依赖修正后的参考解析、主链路 prompt 约束、配饰回填与组合质量门来追求“看起来还是本人”的样本。

## 下一步计划

- 为 `visual_identity` 增加多候选 oversampling 与最佳候选筛选，进一步提升命中率，而不是依赖单次随机采样。
- 继续核查 `hybrid_makeup_refine_stage` 的 `Access Denied` 根因，判断是否可恢复该阶段，或明确将 visual mode 固化为 mainline base candidate 优先。

## 最近修改文件

### [2026-06-28 14:34 Asia/Shanghai] File updates
- 文件名: app/models/job.py
  修改原因: 新增 `identity_mode` 字段，让严格保脸与视觉保本人两种模式可以显式切换。
- 文件名: scripts/run_random_generation.py
  修改原因: 增加 `--identity-mode` 参数并输出 control bundle，便于本地真实跑 visual mode。
- 文件名: app/services/orchestrator.py
  修改原因: 为 visual mode 增加独立质量门、重试路径、区域门控与组合放行条件，并保留 Ark hybrid mainline base candidate 参与最终选择。
- 文件名: app/services/scoring.py
  修改原因: 为 visual mode 调整 identity/transfer/accessory 权重，并提升 hybrid base candidate 的迁移导向评分。
- 文件名: app/services/postprocess.py
  修改原因: 为 visual mode 增加 `accessory_only` 优先选择逻辑，避免眼镜回填被 `none` 模式抢掉。
- 文件名: app/services/reference_parser.py
  修改原因: 修正“盘发+刘海”参考图解析与 region assets 组织方式，使 `reference.png` 被识别为 `updo_with_bangs`。
- 文件名: app/services/model_clients.py
  修改原因: 强化 Ark mainline/hair stage prompt，对 updo、刘海、前额覆盖与脸部结构保持增加更明确约束。

## 当前问题

- 2026-06-27 23:31 Asia/Shanghai: Ark 公开能力面对 identity embedding 和 native mainline executable mask control 仍无官方公开确认，当前仓库继续假设这两项已具备的风险很高。
- 2026-06-27 23:45 Asia/Shanghai: 当前固定样本 `origin2.jpg -> reference7.jpg` 在切到 `two_stage_local_edit` 后，提供方原始结果可达 `identity_score≈0.7043`，但经过任何本地后处理后最高仅到 `0.6257`；说明当前主瓶颈已经进一步收敛到“生成端保脸不够强”，而不是“后处理模式没选对”。

## 当前技术路线

- 2026-06-27 23:31 Asia/Shanghai: P5 结论更新后，当前最稳妥路线应表述为“Ark hybrid mainline + inpaint fallback + 工程控制闭环”，而不是“Ark 原生双分支可执行主链路已确认”。

## 下一步计划

- 基于 P5 结论，优先推进更细粒度结构保护区与本地工程闭环，而不是继续把公开 Ark 主链路假设为已具备原生 identity/mask 控制。
- 如需继续冲击真正 native mainline，需要额外拿到账号侧、产品侧或官方接口侧的更强证据，再决定是否修改 capability probe。
- 下一步优先继续收紧生成前可编辑区，重点减少发型阶段对眼周、鼻梁、唇形、下颌线附近区域的误改，并把 `hair_stage` / `makeup_stage` 分别做独立 identity 观测。

## 最近修改文件

### [2026-06-27 23:31 Asia/Shanghai] File updates
- 文件名: app/models/pipeline.py
  修改原因: 为主链路能力画像补充 `evidence_level`、`confirmed_surfaces`、`missing_surfaces`，明确区分官方确认能力与内部假设能力。
- 文件名: app/services/generator.py
  修改原因: 将 P5 核查结论编码进 `ArkCapabilityProbe`，让系统默认把 Ark 公开主链路判定为 `hybrid` 而非 `native_executable`。

### [2026-06-27 23:45 Asia/Shanghai] postprocess mode selection and scoring-path fix
- 时间: 2026-06-27 23:45 Asia/Shanghai
- 实验名称: postprocess mode selection and scoring-path fix
- 实验配置: 修改 `app/services/postprocess.py` 与 `app/services/scoring.py`，将后处理拆成 `none / accessory_only / light_identity / full_identity` 四档，对同一候选生成多版本并用真实 ArcFace 分数做选择，同时修正评分层在候选已后处理时优先读取 `candidate.image_url` 而不是旧的 `local_output_path`。
- 实验结果: `C:\\Users\\19770\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe -m compileall app` 通过；使用同一解释器运行 `origin2.jpg -> reference7.jpg` 固定样本时，系统自动选中 `selected_postprocess_mode=none`；同一候选的多模式真实身份分分别为 `none=0.6257`、`accessory_only=0.6212`、`light_identity=0.4994`、`full_identity=0.4081`，提供方原始结果 `provider_raw_identity≈0.7043`。
- 实验结论: 后处理多模式选择与评分引用路径已经打通，并证实“更重的本地身份回填会进一步伤害 ArcFace 身份分”；当前默认应避免把 `full_identity` 当作固定后处理，而主瓶颈进一步收敛到生成端本身还不够保脸。

### [2026-06-27 23:45 Asia/Shanghai] File updates
- 文件名: app/services/postprocess.py
  修改原因: 将固定强回填改成多模式后处理与自动模式选择，并记录各模式身份/配饰对比指标。
- 文件名: app/services/scoring.py
  修改原因: 修正候选已后处理时的评分引用路径，避免继续对旧的原始输出文件打身份分。

### [2026-06-28 15:59 Asia/Shanghai] visual accessory refill narrowing and feathered preserve validation
- 时间: 2026-06-28 15:59 Asia/Shanghai
- 实验名称: visual accessory refill narrowing and feathered preserve validation
- 实验配置: 修改 `app/services/postprocess.py`，将 `visual_identity` 的配饰回填范围从整块 `accessory_mask` 收窄为仅保留与 `feature_lock_mask`/脸部上半区相邻的局部配饰区，并把 visual 模式下的配饰保护从硬覆盖改为小半径羽化融合；随后用历史成功任务 `job_bcea76a374bb4722a8f79860d65c33ee` 的 provider 原图 URL 手工重跑后处理验证新选择逻辑，并两次重新执行 `scripts/run_random_generation.py --source origin.jpg --reference reference.png --candidate-count 1 --identity-mode visual_identity`。
- 实验结果: `compileall app` 通过；历史坏样本 `job_438666b54885481495e0d3dace59f192_global_0_hybrid_base_stage_0.png` 中头顶整块原图回贴问题被定位为 visual 模式配饰回填范围过大；手工对历史成功 provider 原图重新运行当前后处理后，`selected_postprocess_mode` 变为 `none`，`preserve_accessory_scope=visual_localized`，不再触发把好图贴坏的 `accessory_only` 选择；两次新的真实重跑分别产生任务 `job_d98f421ca17d4443b97bb13f8f5031e3` 与 `job_26ea0699e2b24f07b9838b2444a38c4c`，两次都在 `ark_hybrid_mainline` global 阶段成功出图，但后续 `hybrid_makeup_refine_stage` / `two_stage_local_edit` 继续遇到 `Access Denied` 与 `cannot identify image file <_io.BytesIO ...>`，最终任务状态均为 `NO_VALID_CANDIDATE`。
- 实验结论: 当前 visual 模式的后处理已从“可能破坏好图”修正为“遇到好图优先不动图”；剩余阻塞重新收敛为 provider 侧 refine/inpaint 不稳定，而不是本地 visual accessory refill 继续错误覆盖头顶盘发区域。

## 当前问题

- 2026-06-28 15:59 Asia/Shanghai: `visual_identity` 后处理误回贴头顶盘发区域的问题已修复，但 provider 侧 `hybrid_makeup_refine_stage` 与 `two_stage_local_edit` 仍频繁返回 `Access Denied`，导致新鲜重跑难以稳定落成成功任务。
- 2026-06-28 15:59 Asia/Shanghai: 手工用本地路径喂 `postprocess_service.run` 做调试时，`load_image_bytes` 仍会把 Windows 路径误判为 base64；真实编排路径不受此问题影响，因为编排时后处理读取的是 provider URL / data URL。

## 下一步计划

- 为 `visual_identity` 增加多候选 oversampling，优先提高 Ark global base candidate 的命中率，减少对 refine/inpaint 成功率的依赖。
- 单独排查 `load_image_bytes` 对 Windows 本地路径的误判，避免后续手工调试再次被路径解析问题干扰。
- 继续核查 `hybrid_makeup_refine_stage` 和 `two_stage_local_edit` 的账号权限 / 返回体稳定性，确认是否需要在 visual 模式下直接跳过这些不稳定阶段。

## 最近修改文件

### [2026-06-28 15:59 Asia/Shanghai] File updates
- 文件名: app/services/postprocess.py
  修改原因: 将 visual 模式的配饰保护收窄到眼镜主导局部区域，并把视觉模式下的配饰保护改成羽化融合，避免把头顶原图头发或脸侧锯齿重新贴回最终图。
- 文件名: PROJECT_MEMORY.md
  修改原因: 追加记录 visual accessory refill 修复、手工后处理复验结果，以及两次新鲜重跑失败仍由 provider refine/inpaint 不稳定导致的事实。

