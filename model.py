# -*- coding: utf-8 -*-

import numpy as np, datetime, inspect, json, shutil, glob
from contextlib import contextmanager
import time, argparse, os, sys, logging, codecs, pickle, pandas as pd, multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from functools import partial


import keras as K
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf,CountVectorizer
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow import set_random_seed
import pandas as pd, numpy as np, pickle
from scipy.sparse import hstack, vstack
import logging, jieba, re
import jieba.posseg as pseg
from collections import defaultdict
from sklearn.metrics import jaccard_similarity_score

from sklearn.model_selection import train_test_split, GridSearchCV,ParameterGrid
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR,SVC
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf,CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

g_new_words = '美团 花呗 摩拜 解绑 拼多多 提额 网商贷 网贷商 双十一 淘票票 余额宝 支付宝 集分宝 天猫 借呗'
g_error_words = '提申:提升,愈期:逾期,花贝:花呗,负款:付款,越期:逾期,途期:逾期,遇期:逾期,马蚁:蚂蚁,花坝:花呗,整么:怎么,网贷商:网商贷,遗期:逾期,提生:提升,提深:提升,伸请:申请'
traditional_chars = "貝車夾見壯妝兒長東岡戔糾來侖兩門協亞軋狀訂訃計飛風負貞釓釔軌軍紇紅紀紉紈紆約紂後勁剄剋則厙帥閂為韋係俠頁唄員狽狹財貢倉倀個倆倫們剗剛芻純紡紛級納紐紕紝紗紓紋紜紖紙島峽峴釘釕釙釗針凍鬥訌記訐訖訕討訊訓迴浹涇郟徑庫馬畝豈氣軔軒孫閃陝陘陣師時書烏挾脅覎隻衹訛訪訣訥訟設訢訩許訝堊堅埡執軛軟區敗絆紬絀紿紱紼紺紹紳細紲組終畢閉參側偉偵從徠釵釧釣釩釹釺釤釷産悵陳陸陰處帶帳頂頃崬崗動務販貫貨貧貪責婦婁婭乾掆捲掄捫掃捨掗剮規覓視匭國圇華莊飢梘條梟將專脛淶淪淺淵連這啢啓問啞鹵麥鳥牽氫殺術烴習現郵魚斬張眾晝硃惡惻悶惱愜惲貳賁貶貸費貴賀貺買貿貰貼貽貯鈀鈈鈔鈄鈍鈁鈣鈎鈥鈞鈧鈉鈕鈐鈦鈃釾報場堝堯備傖傢傘㑳筆殘廁測渢溈渦渾湯湞詞詆詁訶詎評詘訴詗詒詛詐詔診覘硨硤硯棖棟棡棗棧創剴單喬喪喎喲絰給絎絳絞結絕絝絡絨絲統絢腖腎脹隊階陽馱馮馭發復軲軤軻軺軼軫軸媯媧頇順項須壺畫揮揀揚幾間開閔閏閑進痙凱殼睏嵐勞勝勛虜欽韌飥圍幃幀無鄉鄆尋氬猶雲葤愛愴愷愾頒頓頏頎頌頑頊預綁綆經絹綏綈綃綉鉍鉢鈸鉑鈿鈷鉀鈳鈴鉚鉬鈮鈹鉕鉛鉗鈰鉈鉭鉉鈾鈺鉞鉦補裏裊裝滄溝滅溮溳準詫該詬詿詭話詼詰誇誆誄詮詩試詳詡詢詣誅腸腡腦腫馳馴飭飿飯飪飩飲飫傳僅僂僉傾傷傭傴債蒓莢莖莧達過違運當魛搗搶損電煩煉煢煒煬楓極楊業楨鳬鳩賅賈賄賃賂賊資幹嗊嗎嗆嗇嗩嗚號匯彙會琿瑋蛺蜆筧節較輅軾載輊塏塊塒塗塢塤塋壼睞亂媽獁猻獅黽寜農羥義巰肅歲聖勢竪鄔鄖鄒暘暈爺園圓隕閘盞銨鉺銃銚銱銩鉻鉿鉸銬銠銘銓銣銫銅銑銜鉶銥銦銀銪銖骯誒誠誕誥誨誡誑誚認誦說誣誤誘語爾嘔嗶嘗嘍嘜嘆嘖慪慚慘慣慳慶態慟漚滻滌漢滸滬漸漣漊滷滿滲滎漁漲滯漬飽餄餎飼飾餏飴綳綢綽緋綱綸緄緊綾綹綠綿綺綣綬綰網維綫緒綜綻緇綴賓賒賑駁摻摜摑摳摟摶摺暢塵墊塹稱種碭碸碩遞遜遠對奪奬奩罰閥閣閨閡閤閩䦛僨僥僕僑僞瘋瘍鳳鳴鳶韍輔輕輒趕趙構榖榿槍榮幗幘劃夥夢禍禕禎際監盡箋皸匱萊領頗嶁嶇嶄犖屢瑪瑲瑣麼麽齊戧熗熒寢實廎颯颱壽臺圖團窪窩聞覡厭勩與嫗獄殞製皚鋨鋇銼鋤鋒鋯鋏鋦鋃鋰鋁鋩鋪鋟銳鋱銻鋌銷鋅銹鋥噁嘸嘩嘰嘮噴噝嘵嘯餌餃餉養歐毆罷罵鴇鳾鴉鴆輩輥輝輛輪輦輞輜幣幟潷澗漿澆潔潰澇潑潤潿潯潁編締緞緱緩緝緘緙練緬緲緡緦緹緯線緗緣緻標樅槳樂樓槧樞樣樁彆彈撥撣撫擊撟撈撓撲撳撏摯層賜賧賦賡賫賤賚賣賠䝼 賞賢賬質賙嬋嫿嬌嬈嫵嫻諂調誹課諒論請誶誰諗談諉誼諏諍諄諑廠廢廣廟廡徹徵齒衝衛䦟閫 閬閭閱瘡瘧瘞鄲鄧鄰鄭憚憤憒憐慮憫慤慫憮憲憂導敵數墮墳墜髮範篋魴魯魷膚膕膠膞麩複駙 駕駒駑駟駛駘駝駔駐頜頡頦鞏颳劊劌劍劇劉葒葷蕒萬葦萵葉翬價儉儈儂儀億踐嶠嶗嶔嶢靚寬 審寫厲璉瑩碼確磑鬧盤窮熱毿殤蝕蝸蝦適豎興暫皺錒錛錶錯錘鍀錠鋼錮錦鋸錁錕錸錄鍆錳鍩 錇錢錫鍁錚錙錐噯噠噹噸噲駡噥噦嬡嬙諳諤諶諦諜諷諱諢諫謀諾諞謂諧諝諼諺謁諛諭諮諸餓 餑餜餒餘甌鮁鮑鮒鮎鮍鮃鮐鮣鮓辦憊懇憑憶懌應儐儕儔儘穇積穌穎艙蒼蓋蓀蒔輳輻輯輸縗縛 縞縑縉縟縧縣縊縈縕縝縐縋閶閽閿閹閼閻閾磣磧磚赬鴟鴣鴒鴝鴕鴞鴨鴦鴛遲遼遷選遺熾燈燜 燒燙燁擔擋撿據擓擄撻撾擁擇殫殨澱澮澠濃澾澦澤濁窵窶窺獨獪獫篤篩築賭賵賴奮龜駭駱駢 頷頰頸頻頲頭頹頤樺機樸橋橈樹橢璣璡劑舉墾壇瞘瞜瞞褲鄶鄴曆曇曉曄歷龍瘻瘮盧螞螄螢膩 親覦隨險學嶨嶧踴禦醞戰鎄鍔鍤鍍鍛鍋鍰鍵鍇錨鎇鎂鍬鍥鍶鍚鍘鍺鍾曖璦環鴯鴿鴰鴴鴻鵁鴷 鵂鵃闆闊闌闃闋闈幫謗謊講謎謐謙謖謚謄謝謔謠謅篳簍簀蓽蔔蓯蔣蓮蔞蔦蔭濱濟濜濫濛濘澀 濕濤濰擯擱擠擬擰餅餷餶館餛餞餡燦燴營燭聰聯聳聲禪償優齔檉檔檜檟檢檣騁駿駸醜膽膾臉 膿癉癆療癇點糞糝縫績縲縭縷縵繆縹縴繅縮總縱縶賻購賽賺尷覯覬轂轄輿轅輾鮭鮫鮚鮦鮪鮮 鮝鮺蟈螻蟎蟄還邁韓嚇嚀獲獰磯磽艱襇襉褳褸褻矯屨颶顆虧勵隸斂殮瞭臨嶺嶸嶼耬懞懨彌嬪 嬰牆蹌趨麯雖戲壓隱黿齋氈額題顔顒顓鵝鵓鵠鵑鵜鵡鵒邇謳謭謹謾謬謨謫擺擷擴攆擾擻攄擲 鎊鎬鎘鎵鎸鎧鎦鎿鎳鎪鎖䥇鎢鎰鎮鎡斃蹕蹣檳櫃檻檸檯殯臏臍叢竄竅蟬蟲蟣蟯蕆蕩蕢蕁蕎蕘 蕪蕭蕕蕓雛雞雙雜儲礎闖闔闓闕闐簞簡簣簫鼕瀆濺瀏濼濾瀋瀉斷懟懣豐鯁鯀鯇鯉鯊獷獵歸 穢穡瞼醬醫燼燾覲舊騍騎騏騅壙壘鄺䙡聵聶職耮禮癘糧魎繚繞繕繒織嚕嚙轆轉翹鞦軀鬆觴嬸 韙鬩餳歟賾贄贅礙鵪鶉鶇鶊鶄鶓鵬鵮鵲襖襠襝鏰鏢鏟鏑鏡鏗鏈鏤鏝鏘鏹鎩鏜鏇鏞鏨鏃邊癟瀕 瀨瀝瀧瀟餺餾饃餿餼辭蠆蟶蟻蠅鯧鯛鯡鯨鯤鯪鯢鯰鯖鯫鯔懲懷懶寵疇禱禰顛類顙願櫝櫟櫓櫚 櫞櫛牘犢關鬍壞壢壟壚繯繪繫繭繮繳繰繩繡繹薈薊薦薑薔薌蕷穫穩譏譎譜譙識譚譖證轎轔轍 曠臘離難嚦嚨嚮麗簾簽飀颼攏隴廬氌羅羆矇龐騙騖蹺蹻瓊璽覷勸獸獺獻爍韜韞霧厴贋贊贈鰐 鯿鰆鰈鰒鰉鯽鰌鰍鰓鰂鶚鶿鷀鶡鶘鶪鶥鶖鶩齙齣齟齡齠齜寶辮繽繼繾攙攔攖闡闞懺懸櫬櫪櫳 櫨櫧籌籃觸黨鐙鐓鐨鏵鐧鐫鐦鐒鐐鐃鏺鏷鐠鐥鐘鐯竇躉礬礦礪礫豶薺藉藎藍薩艦覺癤癢癥饉 饅饊饈競嚳嚴嚶巋瀾瀲瀰襤騮騫騷騸騰騶騭曨朧瓏爐臚顢獼麵聹飄蠐蠑贍贏釋糰犧鹹響龑譯 議譫罌驁驃驂驄騾驀驅䎱襬襪辯飆鶬鶻鶴鷄鶺鶼鷉鷂鶯攛㩳攝鹺纏纊纍纈續鐺鐸鐶鐳鐮鐵鐿 鐲躊躋躍灃灄顧顥鰥鰜鰟鰭鰣鰨鰧鰮䱷轟護譴譽齎殲贐懼懾櫸欄櫻饋饒饌蠟蠣癩癧斕覽爛儷 儺糲礱髏衊囁囂囀闢闥鞽藭藪藥藝屬齦瓔竈鰲鰾鰹鰳鰱鰻鰵鰷鰼鱈鱅鷗鷚鷖鷓鷙鑌鑔鑊鑒鑄 攢攤囅囉囈顫襯襲齪齬韃韁糴覿巔巒讀讅龔龕驊驕驍歡霽籟籠籙籜灕灑灘藶藺蘢蘆蘋蘄蘇蘀 蘊轢轡酈聾聽艫孌孿玀竊權贖罎儻儼彎饗蠨鬚癬癮癭躑躓變讎讌讋鑣鑞鑥鑠鑕鱉鱖鱗鱔鱘鱒 纔纖纓蠱㘚囌齏鷦鷲鷯鷺鷥鷳鷸攪攣驚驗驛蘭蘞蘚邐邏戀轤欒欏黴籤顬顯曬體髒躚饜靨癰臢 瓚靄靆靂靈壩髕鬢蠶讒讖讕讓癲癱籪贛灝鱟鱠鱧鱣羈鹼鹽攬釀鷿鸊鷹鸇顰韆齲齷艷驟囑靉籩 籬籮躥躡鱨鱭觀黌髖鑭鑲鑰欖顱臠蘿蠻糶廳鼉灣釁纘鑹鑷饞韉灤驢釃黶趲躦矚鱷鱸讜讞黷驥 驤纜躪鸕鑾鑼鑿鑽顳顴灧釅鐝钁钂欞鸚戇鸛驪鬱鱺鸝鸞饢籲膁鬨瀠颻襆鏍閎憖丟眥氳漵媼饑鯒齕虯瘺湊奐緓寧薘輟鰩脫蓴鏌貟鮋瑤輈琺兌鍠廩撚羨闤奧 棲鋶餱鯕鉤翽鐋嚌溜缽訁膃搖頴輇袞別佇闠齧詵掙產屜嶽雋闍獎鋣餕減堿著煥郤銍鎣遝镟禿 魨襴梔贇閌貲潛癡哢偽祿騌敘谘茲鳲槁壋魢柵巹鏽鸏繃筍彥糸颭傑鈒夠瀦阪鯷蒞墶簹鈽汙蝟 囪鐔悅璿嫋姍穠櫫勻鰏瘓唕鵐鮶噓淚橫並棬檁暉亙掛薟矓詖佘換鈑渙鶤鶲毀晉絏鏐內卻災鸌 鋙鰠騤馹荊颸魺矽讚爿驏緶脈箏褘崠嬤刪盜況顎繢鎔強嘽懍灩腳脊崢溫靦豬齶呼萇崍嚐閒錩 冪簷咼棄闒虛羋輬吒攜轀瑒彠韻壪綌贗玨檮怵錡屍鵷貓靜鯝淩藹綞鷊摣訒躕醱蘺鶹贓鮜爭崳 噅驌濔淨駰唚澩繈拋豔硜娛喚蔥紵鼴廈慍贔粵戶賕弳兗扡滾纇瀘璫艤驦劄鮊稟煙飣湧醃瘂屆 廝俁鼇躒齇秈遙絛槨黲鯴歿稈蠔傯嶴臥獮艸譾臟猙鮞榪糊淒緔鮳呂軑鞀瀅沒甕涼檾鱝恥嗬餖 鋝紮垵煆蹤苧鰃誌嶮吳饁廄顏觶騂麅堖紘螿鯗裡閈氹瑉躂屙廚剝棶錆鸘犛黃詠淥鬁疊熲鯵臒 湣驫鱯冇隉褌鎛綯璵諡勱軹決穭蠍鱂猛凜啟鵯鼂窯洶鷁燉韝跡幬閆撐銛譸愨鳧頮睜櫥鷫異颮 逕鰁稅穀錟嚲廂齗呱覥鬮歎鱤蹠颺蓧勳蛻劈冊钜宮皰侶擼礄錈鴬鍈遊魘塚坰椏戩於襏屭"
simple_chars = "贝车夹见壮妆儿长东冈戋纠来仑两门协亚轧状订讣计飞风负贞钆钇轨军纥红纪纫纨纡约纣后 劲刭克则厍帅闩为韦系侠页呗员狈狭财贡仓伥个俩伦们刬刚刍纯纺纷级纳纽纰纴纱纾纹纭纼 纸岛峡岘钉钌钋钊针冻斗讧记讦讫讪讨讯训回浃泾郏径库马亩岂气轫轩孙闪陕陉阵师时书 乌挟胁觃只只讹访诀讷讼设䜣讻许讶垩坚垭执轭软区败绊䌷绌绐绂绋绀绍绅细绁组终毕闭参 侧伟侦从徕钗钏钓钒钕钎钐钍产怅陈陆阴处带帐顶顷岽岗动务贩贯货贫贪责妇娄娅干㧏卷抡 扪扫舍挜剐规觅视匦国囵华庄饥枧条枭将专胫涞沦浅渊连这唡启问哑卤麦鸟牵氢杀术烃习现 邮鱼斩张众昼朱恶恻闷恼惬恽贰贲贬贷费贵贺贶买贸贳贴贻贮钯钚钞钭钝钫钙钩钬钧钪钠钮 钤钛钘䥺报场埚尧备伧家伞㑇笔残厕测沨沩涡浑汤浈词诋诂诃讵评诎诉诇诒诅诈诏诊觇砗硖 砚枨栋㭎枣栈创剀单乔丧㖞哟绖给绗绛绞结绝绔络绒丝统绚胨肾胀队阶阳驮冯驭发复轱轷轲 轺轶轸轴妫娲顸顺项须壶画挥拣扬几间开闵闰闲进痉凯壳困岚劳胜勋虏钦韧饦围帏帧无乡郓 寻氩犹云荮爱怆恺忾颁顿颃颀颂顽顼预绑绠经绢绥绨绡绣铋钵钹铂钿钴钾钶铃铆钼铌铍钷铅 钳铈铊钽铉铀钰钺钲补里袅装沧沟灭浉涢准诧该诟诖诡话诙诘夸诓诔诠诗试详诩询诣诛肠脶 脑肿驰驯饬饳饭饪饨饮饫传仅偻佥倾伤佣伛债莼荚茎苋达过违运当鱽捣抢损电烦炼茕炜炀枫 极杨业桢凫鸠赅贾贿赁赂贼资干唝吗呛啬唢呜号汇汇会珲玮蛱蚬笕节较辂轼载轾垲块埘涂坞 埙茔壸睐乱妈犸狲狮黾宁农羟义巯肃岁圣势竖邬郧邹旸晕爷园圆陨闸盏铵铒铳铫铞铥铬铪铰 铐铑铭铨铷铯铜铣衔铏铱铟银铕铢肮诶诚诞诰诲诫诳诮认诵说诬误诱语尔呕哔尝喽唛叹啧怄 惭惨惯悭庆态恸沤浐涤汉浒沪渐涟溇卤满渗荥渔涨滞渍饱饸饹饲饰饻饴绷绸绰绯纲纶绲紧绫 绺绿绵绮绻绶绾网维线绪综绽缁缀宾赊赈驳掺掼掴抠搂抟折畅尘垫堑称种砀砜硕递逊远对夺 奖奁罚阀阁闺阂合闽䦶偾侥仆侨伪疯疡凤鸣鸢韨辅轻辄赶赵构谷桤枪荣帼帻划伙梦祸祎祯际 监尽笺皲匮莱领颇嵝岖崭荦屡玛玱琐么么齐戗炝荧寝实庼飒台寿台图团洼窝闻觋厌勚与妪狱 殒制皑锇钡锉锄锋锆铗锔锒锂铝铓铺锓锐铽锑铤销锌锈锃恶呒哗叽唠喷咝哓啸饵饺饷养欧殴 罢骂鸨䴓鸦鸩辈辊辉辆轮辇辋辎币帜滗涧浆浇洁溃涝泼润涠浔颍编缔缎缑缓缉缄缂练缅缈缗 缌缇纬线缃缘致标枞桨乐楼椠枢样桩别弹拨掸抚击挢捞挠扑揿挦挚层赐赕赋赓赍贱赉卖赔䞍 赏贤账质赒婵婳娇娆妩娴谄调诽课谅论请谇谁谂谈诿谊诹诤谆诼厂废广庙庑彻征齿冲卫䦷阃 阆闾阅疮疟瘗郸邓邻郑惮愤愦怜虑悯悫怂怃宪忧导敌数堕坟坠发范箧鲂鲁鱿肤腘胶䏝麸复驸 驾驹驽驷驶骀驼驵驻颌颉颏巩刮刽刿剑剧刘荭荤荬万苇莴叶翚价俭侩侬仪亿践峤崂嵚峣靓宽 审写厉琏莹码确硙闹盘穷热毵殇蚀蜗虾适竖兴暂皱锕锛表错锤锝锭钢锢锦锯锞锟铼录钔锰锘 锫钱钖锨铮锱锥嗳哒当吨哙骂哝哕嫒嫱谙谔谌谛谍讽讳诨谏谋诺谝谓谐谞谖谚谒谀谕谘诸饿 饽馃馁余瓯鲅鲍鲋鲇鲏鲆鲐䲟鲊办惫恳凭忆怿应傧侪俦尽䅟积稣颖舱苍盖荪莳辏辐辑输缞缚 缟缣缙缛绦县缢萦缊缜绉缒阊阍阌阉阏阎阈碜碛砖赪鸱鸪鸰鸲鸵鸮鸭鸯鸳迟辽迁选遗炽灯焖 烧烫烨担挡捡据㧟掳挞挝拥择殚㱮淀浍渑浓㳠滪泽浊窎窭窥独狯猃笃筛筑赌赗赖奋龟骇骆骈 颔颊颈频颋头颓颐桦机朴桥桡树椭玑琎剂举垦坛眍䁖瞒裤郐邺历昙晓晔历龙瘘瘆卢蚂蛳萤腻 亲觎随险学峃峄踊御酝战锿锷锸镀锻锅锾键锴锚镅镁锹锲锶钖铡锗钟暧瑷环鸸鸽鸹鸻鸿䴔䴕 鸺鸼板阔阑阒阕闱帮谤谎讲谜谧谦谡谥誊谢谑谣诌筚篓箦荜卜苁蒋莲蒌茑荫滨济浕滥蒙泞涩 湿涛潍摈搁挤拟拧饼馇馉馆馄饯馅灿烩营烛聪联耸声禅偿优龀柽档桧槚检樯骋骏骎丑胆脍脸 脓瘅痨疗痫点粪糁缝绩缧缡缕缦缪缥纤缫缩总纵絷赙购赛赚尴觏觊毂辖舆辕辗鲑鲛鲒鲖鲔鲜 鲞鲝蝈蝼螨蛰还迈韩吓咛获狞矶硗艰裥裥裢褛亵矫屦飓颗亏励隶敛殓了临岭嵘屿耧蒙恹弥嫔 婴墙跄趋曲虽戏压隐鼋斋毡额题颜颙颛鹅鹁鹄鹃鹈鹉鹆迩讴谫谨谩谬谟谪摆撷扩撵扰擞摅掷 镑镐镉镓镌铠镏镎镍锼锁䦂钨镒镇镃毙跸蹒槟柜槛柠台殡膑脐丛窜窍蝉虫虮蛲蒇荡蒉荨荞荛 芜萧莸芸雏鸡双杂储础闯阖闿阙阗箪简篑箫冬渎溅浏泺滤沈泻断怼懑丰鲠鲧鲩鲤鲨犷猎归 秽穑睑酱医烬焘觐旧骒骑骐骓圹垒邝䙌聩聂职耢礼疠粮魉缭绕缮缯织噜啮辘转翘秋躯松觞婶 韪阋饧欤赜贽赘碍鹌鹑鸫鹒䴖鹋鹏鹐鹊袄裆裣镚镖铲镝镜铿链镂镘锵镪铩镗旋镛錾镞边瘪濒 濑沥泷潇馎馏馍馊饩辞虿蛏蚁蝇鲳鲷鲱鲸鲲鲮鲵鲶鲭鲰鲻惩怀懒宠畴祷祢颠类颡愿椟栎橹榈 橼栉牍犊关胡坏坜垄垆缳绘系茧缰缴缲绳绣绎荟蓟荐姜蔷芗蓣获稳讥谲谱谯识谭谮证轿辚辙 旷腊离难呖咙向丽帘签飗飕拢陇庐氇罗罴蒙庞骗骛跷硚琼玺觑劝兽獭献烁韬韫雾厣赝赞赠鳄 鳊䲠鲽鳆鳇鲫䲡鳅鳃鲗鹗鹚鹚鹖鹕䴗鹛鹙鹜龅出龃龄龆龇宝辫缤继缱搀拦撄阐阚忏悬榇枥栊 栌槠筹篮触党镫镦镄铧锏镌锎铹镣铙䥽镤镨䦅钟䦃窦趸矾矿砺砾豮荠借荩蓝萨舰觉疖痒症馑 馒馓馐竞喾严嘤岿澜潋弥褴骝骞骚骟腾驺骘昽胧珑炉胪颟猕面聍飘蛴蝾赡赢释团牺咸响䶮译 议谵罂骜骠骖骢骡蓦驱䎬摆袜辩飙鸧鹘鹤鸡鹡鹣䴘鹞莺撺㧐摄鹾缠纩累缬续铛铎镮镭镰铁镱 镯踌跻跃沣滠顾颢鳏鳒鳑鳍鲥鳎䲢鳁䲣轰护谴誉赍歼赆惧慑榉栏樱馈饶馔蜡蛎癞疬斓览烂俪 傩粝砻髅蔑嗫嚣啭辟闼鞒䓖薮药艺属龈璎灶鳌鳔鲣鳓鲢鳗鳘鲦鳛鳕鳙鸥鹨鹥鹧鸷镔镲镬鉴铸 攒摊冁啰呓颤衬袭龊龉鞑缰籴觌巅峦读谉龚龛骅骄骁欢霁籁笼箓箨漓洒滩苈蔺茏芦苹蕲苏萚 蕴轹辔郦聋听舻娈孪猡窃权赎坛傥俨弯飨蟏须癣瘾瘿踯踬变雠䜩詟镳镴镥铄锧鳖鳜鳞鳝鲟鳟 才纤缨蛊㘎苏齑鹪鹫鹩鹭鸶鹇鹬搅挛惊验驿兰蔹藓逦逻恋轳栾椤霉签颥显晒体脏跹餍靥痈臜 瓒霭叇雳灵坝髌鬓蚕谗谶谰让癫瘫簖赣灏鲎鲙鳢鳣羁硷盐揽酿䴙䴙鹰鹯颦千龋龌艳骤嘱叆笾 篱箩蹿蹑鲿鲚观黉髋镧镶钥榄颅脔萝蛮粜厅鼍湾衅缵镩镊馋鞯滦驴酾黡趱躜瞩鳄鲈谠谳黩骥 骧缆躏鸬銮锣凿钻颞颧滟酽镢䦆镋棂鹦戆鹳骊郁鲡鹂鸾馕吁肷闹潆飖幞镙闳慭丢眦氲溆媪饥鲬龁虬瘘凑奂绬宁荙辍鳐脱莼镆贠鲉瑶辀珐兑锽廪捻羡阛奥 栖锍糇鲯钩翙铴哜熘钵讠腽摇颕辁衮别伫阓啮诜挣产屉岳隽阇奖铘馂减碱着焕郄铚蓥沓碹秃 鲀襕栀赟闶赀潜痴咔伪禄骔叙咨兹鸤藁垱鱾栅卺锈鹲绷笋彦纟飐杰钑够潴坂鳀莅垯筜钸污猬 囱镡悦璇袅姗秾橥匀鲾痪唣鹀鲪嘘泪横并桊檩晖亘挂莶眬诐畲换钣涣鹍鹟毁晋绁镠内却灾鹱 铻鳋骙驲荆飔鲄硅赞丬骣缏脉筝袆岽嬷删盗况颚缋镕强啴懔滟脚嵴峥温腼猪腭唿苌崃尝闲锠 幂檐呙弃阘虚芈辌咤携辒玚彟韵塆绤赝珏梼憷锜尸鹓猫静鲴凌蔼缍鹝揸讱蹰酦蓠鹠赃鲘争嵛 咴骕沵净骃吣泶襁抛艳硁娱唤葱纻鼹厦愠赑粤户赇弪兖扦滚颣泸珰舣骦札鲌禀烟饤涌腌痖届 厮俣鳌跞齄籼遥绦椁黪鲺殁秆蚝偬岙卧狝艹谫脏狰鲕杩煳凄绱鲓吕轪鼗滢没瓮凉苘鲼耻呵饾 锊扎埯煅踪苎鳂志崄吴馌厩颜觯骍狍垴纮螀鲞里闬凼珉跶疴厨剥梾锖鹴牦黄咏渌疬叠颎鲹癯 愍骉鳠没陧裈镈绹玙谥劢轵决稆蝎鳉勐凛启鹎鼌窑汹鹢炖鞴迹帱闫撑铦诪悫凫颒睁橱鹔异飑 迳鳈税谷锬亸厢龂哌觍阄叹鳡跖飏莜勋蜕噼册鉅宫疱侣撸硚锩鸴锳游魇冢垧桠戬于袯屃" 

traditional_chars = list(traditional_chars.replace(' ',''))
simple_chars = list(simple_chars.replace(' ',''))
T2S = dict(zip(traditional_chars, simple_chars))
S2T = dict(zip(simple_chars, traditional_chars))


TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tf.logging.set_verbosity(tf.logging.INFO)

from numpy.random import seed
seed(1)
set_random_seed(2)
start_time = time.time()

gl = globals()    
if os.path.exists('data'):
    g_local = True
    data_dir = 'data'
else:
    g_local = False
    data_dir = model_dir

def multiple_replace(dict ):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    def replace(text):
        return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)
    return replace

g_debug_cnt = 500
g_trans = {}
g_result = {}


def load_data(fpath = None, random = True, use_local=False):
    if fpath is None:
        fpath = 'data/atec_nlp_sim_train_all.csv'
    #df1 = pd.read_csv('data/atec_nlp_sim_train_add.csv',error_bad_lines=False, sep='\t', header = None)
    #df2 = pd.read_csv('data/atec_nlp_sim_train.csv',error_bad_lines=False, sep='\t', header = None)
    #df = pd.concat([df1,df2])
    if g_local or use_local:
        df = pd.read_csv(fpath, error_bad_lines=False, sep='\t', header = None)
    else:
        df = df1
    logging.info("total %s rows", len(df))

    if random:
        df = df.sample(frac=1, random_state = 94).reset_index(drop=True)
    if len(df.columns) ==4:
        df.columns = ['id','s1','s2','y']
    else:
        df.columns = ['id','s1','s2']

    for col in ['s1','s2']:
        df[col] = df[col].str.lower().str.replace(' +', ' ').str.strip()
    #if args.t2s:
    if 1==1:
        mp = multiple_replace(T2S)
        #mp2 = multiple_replace(trans)
        for col in ['s1','s2']:
            sents = df[col].values; pys = []
            for sent in sents:
                sent = mp(sent)
                #sent = mp2(sent)
                #sent = re.sub('[*]+','dstars',sent)
                pys.append(sent)
            df[col] = pys
    return df

def tok_jieba(s):
    return list(jieba.cut(s))

def tok(s):
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+',s)
    return words


def texts_to_sequences(sents, word_index, tok, max_len):
    seqs = []
    unk = set()
    for s in sents:
        #s = s.decode('utf-8')
        word_seq = tok(s)

        seq = [1]#1:sos 0:eos 2:unk
        for w in word_seq:
            if w in word_index:
                seq.append(word_index[w])
            else:
                unk.add(w)
                #seq.append(word_index['UNK'])
                
        seqs.append(seq)
        #seqs.append(seq[0:max_len])
    logging.info('total %s new words:%s', len(unk), '|'.join(list(unk)[0:10]))
    return seqs
def seq(dT, dV, max_len, word_index = None, min_cnt = 5, embedding_fpath = 'data/anti_w2v_model', data_dir = data_dir, tokenizer = tok_jieba):
    logging.info('start seq data')
    fcnt = 'word_cnt.dump'
    if args.char:
        tokenizer = tok
        fcnt = 'char_' + fcnt
    elif args.tok:
        tokenizer = gl[args.tok]
        fcnt = args.tok +'_' + fcnt
    if dT is not None:
        numT = len(dT)

        #tok = Tokenizer(char_level=True)
        sents = pd.concat([dT.s1, dT.s2])
        word_index = {};ind = 3; word_cnt = defaultdict(int)
        utf8_sents = []; seen_sents = set()
        for s in sents:
            if s in seen_sents:
                continue
            #s = s.decode('utf-8')
            utf8_sents.append(s)
            seq = tokenizer(s)
            for w in seq:
                word_cnt[w] += 1
            seen_sents.add(s)
        logging.info("all word cnt:%s", len(word_cnt))
        for key, v in sorted(word_cnt.items()):
            if v  >= min_cnt:
                word_index[key] = ind
                ind += 1
        word_index['UNK'] = 2
        word_index['SOS'] = 1
        word_index['EOS'] = 0
        logging.info("word_index num:%s", len(word_index))
        if embedding_fpath:
            embeddings = load_embeddings(word_index, embedding_fpath, data_dir = data_dir)
        else:
            logging.info("not use w2v embeddings")
            embeddings = None
        seqs = texts_to_sequences(sents, word_index, tok = tokenizer, max_len=max_len)
        seqs_len = np.array(list(map(lambda x: min(x, max_len-1), map(len, seqs)))) + 1
        logging.info("max len for train seq is %s", max(map(len, seqs)))

        xT = pad_sequences(seqs, max_len, truncating='post', padding='post')
        xT = {'s1': xT[0:numT, :],
              's2': xT[numT:,:],
              's1_seqs_len': seqs_len[0:numT],
              's2_seqs_len': seqs_len[numT:],
               'ids': dT.id.values,
             }
        if 'ex_feas' in dT.columns:
            xT['ex_feas'] = np.array(list(dT['ex_feas'].values))
        yT = dT.y.values
    else:
        xT = None
        yT = None
        embeddings = None
        word_cnt = None
    numV = len(dV)
    seqs_s1 = texts_to_sequences(dV.s1, word_index, tok = tokenizer, max_len=max_len)
    seqs_s2 = texts_to_sequences(dV.s2, word_index, tok = tokenizer, max_len = max_len)
    logging.info("max len for validate seq is %s", max(map(len, seqs_s1 + seqs_s2)))
    xV = {
            's1': pad_sequences(seqs_s1, max_len, truncating='post', padding='post'),
            's2': pad_sequences(seqs_s2, max_len, truncating='post', padding='post'),
            's1_seqs_len': np.array(list(map(lambda x: min(x, max_len-1),map(len, seqs_s1)))) +1,
            's2_seqs_len': np.array(list(map(lambda x: min(x, max_len-1),map(len, seqs_s2)))) +1,
            'ids': dV.id.values,
          }
    if 'ex_feas' in dV.columns:
        xV['ex_feas'] = np.array(list(dV['ex_feas'].values))
    yV = dV.y.values

    return xT, xV, yT, yV, dT, dV, word_index, embeddings, word_cnt

def df2dic(df):
    columns = df.columns
    keys = df['key'].values
    values = df[columns[0:-1]].values
    w2v = dict(zip(keys, values))
    return w2v
def load_embeddings(word_index, fname='w2v.csv', data_dir = data_dir):
    #if g_local:
    fname = os.path.join(data_dir, fname)
    df = pd.read_csv(fname)
    #else:
    #    df = df2
    #wv = gensim.models.Word2Vec.load(fpath).wv
    #emb_dim = wv.vectors.shape[1]
    wv = df2dic(df)
    logging.info('w2v word num is %s', len(wv))
    for key, value in wv.items():
        emb_dim = len(value)
        break
    embeddings = np.zeros([len(word_index), emb_dim])

    i = 0
    unk = []
    for w,v in sorted(word_index.items()):
        if w in wv:
            if word_index[w] == len(word_index):
                logging.info('index error for word:%s', w)
            embeddings[word_index[w]] = wv[w]
        else:
            embeddings[word_index[w]] = np.random.rand(emb_dim)
            #embeddings[word_index[w]] = np.zeros(emb_dim)
            unk.append(w)
            i += 1
    logging.info('load embedding: %s total %s new words:%s',fname, i, '|'.join(unk))
    return embeddings.astype(np.float32)



def w2v(fname=None, outname='w2v.csv', emb_dim = 64, vp = None, df = None,  data_dir = data_dir, tokenizer = tok_jieba, seed = 1, min_cnt = None, workers = multiprocessing.cpu_count() ):
    dV = None
    if min_cnt is None:
        min_cnt = args.mc
    if vp is None:
        vp = args.vp
    if args.emb_dim:
        emb_dim = args.emb_dim
    if args.input is not None:
        fname = args.input
    if args.output is not None:
        outname = args.output
    if args.char:
        tokenizer = tok
        outname = 'char_' + outname
    elif args.tok:
        tokenizer = gl[args.tok]
        outname = args.tok +'_' + outname

    outname = str(emb_dim) + 'd_' + outname    
    #if args.debug:
    #    outname = 'debug_' + outname
        
        
    if fname is None:
        if df is None:
            df = load_data()
            if args.debug:
                df = df[0:g_debug_cnt]
            df, dV = train_test_split(df, test_size = vp, random_state=np.random.RandomState(8763))
        else:
            dV = None
    
        sents = pd.concat([df.s1,df.s2]).unique()
        inp = []
        for sent in sents:
            #line = tokenizer(re.sub(' +',' ', sent))
            line = tokenizer(sent)
            inp.append(line)

    model = Word2Vec(inp, size=emb_dim, window=10, min_count=min_cnt, iter=5, seed = seed,
                     workers=workers)
    wv = model.wv
    keys = []
    values = []
    for key in wv.vocab:
        keys.append(key)
        values.append(wv[key])
    df = pd.DataFrame(np.array(values))
    df['key'] = keys
    
    fname = os.path.join(data_dir, outname)
    df.to_csv(fname, index=False)
    if dV is not None:
        save_validate(dV, vp = vp)

    if not g_local and dV is not None:
        topai(1, dV)
    return fname

def search_pred_thr(pred, y = None):
    if y is None:
        y = load_validate().y.values

    #pred = pd.read_csv(os.path.join(data_dir, 'test_pred_prob.csv'))
    #pred.columns = ['id','y']
    best_score = -10000.0; best_pred_thr = None
    for pred_thr in np.arange(0.01,0.99, 0.01):
        score = f1_score(pred>pred_thr, y)
        if score > best_score:
            best_score = score
            best_pred_thr = pred_thr
    logging.info('best pred thr is %s, score is %s', best_pred_thr, best_score)
    return best_pred_thr, best_score
        




    


class Model(object):
    seed = 8763
    _layers = [128]
    _lr = 1e-3
    _epochs = None
    _batch_size = 2000
    _max_batch_size = 4000
    _lr_decay_rate = 1.0
    #_pred_thr = 0.18233
    _pred_thr = 0.25
    _emb_fname = 'w2v.csv'
    _emb_dim = 64
    if not g_local:
        _pred_thr = 0.35
    _es = 0.03
    def __init__(self, name = None, **kwargs):
        self.name = name
        if self.name is None:
            self.name = self.__class__.__name__
        #if args.char:
        #    self.name = self.name + '_CHAR'
        elif args.tok:
            self.name = self.name + '_' + args.tok
        self._rs = np.random.RandomState(self.seed)
        self._model = None
        if g_local:
            #self._timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self._timestr = ''
        else:
            self._timestr = ''
        if args.model_name=='eval_model':
            self._timestr = ''
#        with codecs.open('stop_words.txt', encoding='utf-8') as f:
#            stopwords = f.read()
#        self._stop_words = []
#        for w in stopwords.split(' '):
#            self._stop_words.append(w)
        self.data_dir = data_dir
        if args.save:
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
    def _create_model(self, inShape):
        pass
    def show(self,df, pred, y, x):
        for i in show(df, pred, y):
            yield
    def load_preds(self):
        fname = 'pred.csv'
        pred = pd.read_csv(self._gen_fname('', fname)).pred.values
        return [pred]
    def _save_predict(self, predV, validation_data):
        df = pd.DataFrame()
        df['pred'] = predV
        df['y'] = validation_data[1]
        fname = 'pred.csv'
        df.to_csv(self._gen_fname('', fname), index=False)
    def fit(self, x, y, validation_data = None, save = True):
        


        predV = self.predict(validation_data[0])
        if save:
            self.save()
            self._save_predict(predV, validation_data)
        return predV
    def _gen_fname(self, postfix, *paras):
        if args.debug:
            name = os.path.join(self.data_dir, 'debug_' + self.name + self._timestr + '_' + postfix, *paras)
        else:
            name = os.path.join(self.data_dir, self.name + self._timestr + '_' + postfix, *paras)
        return name
    def save(self):
        fpath = self._gen_fname('', 'paras.txt')
        dirname = os.path.dirname(fpath)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if g_local:
            shutil.copy2('models.py', dirname)
            shutil.copy2('util.py', dirname)

    def predict(self, x):
        pred = self._model.predict(x)
        if len(pred.shape) >1:
            pred = pred[:,0]
        return pred

    @classmethod
    def f1_score(cls, y, pred):
        tp = np.sum((y==pred)[pred==1])
        fp = np.sum((y!=pred)[pred==1])
        fn = np.sum((y!=pred)[pred==0])
        precision = tp/(tp+fp + 1e-15) 
        recall = tp/(tp+fn + 1e-15)
        score = 2*precision*recall/(precision+recall + 1e-15)
        logging.info('tp:%s, fp:%s, fn:%s, precision:%s, recall:%s, score:%s', tp, fp, fn, precision, recall, score)
        return score

    def score(self, x, y, thr=0.25):
        pred = self.predict(x)
        pred = pred > thr
        score = self.f1_score(y, pred)
        return score

    def format_data(self, df, pctV = 0.05, dT = None, dV = None):
        if df is not None:
            if pctV is not None:
                dT, dV = train_test_split(df, test_size = pctV, random_state= np.random.RandomState(8763))
                if args.ext:
                    #dT = pd.read_csv('data/ext_train.csv')
                    dT_ext = pd.read_csv('data/ext_train.csv')
                    dT_ext['s1']=dT_ext.s1.str.lower()
                    dT_ext['s2']=dT_ext.s2.str.lower()
                    logging.info('******info*****ext rows is %s', len(dT))
                    #dT = pd.concat([dT,dT_ext])
                if args.debug:
                    dT = dT[0:g_debug_cnt]
            else:
                dT = None
                dV = df
        #self.dT = dT; self.dV = dV
        if dT is not None:
            logging.info('train rows:%s', len(dT))
        if dV is not None:
            logging.info('valid rows:%s', len(dV))
        return dT, dV
    def format_eval(self, df):
        return df 

def layer(fuc):
    class Layer(object):
        def __init__(self, *args, **kwargs):
            self._fuc = fuc
            self.args = args
            self.kwargs = kwargs
            self.name = self.kwargs.get("name", self._fuc.__name__)

            self._template = tf.make_template(self.name, self._fuc, create_scope_now_ = True)

        def __call__(self, x):
            out = self._template(x, *self.args, **self.kwargs)
            #self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return out

        def __rrshift__(self, other):
            """ >> """
            return self.__call__(other)
    return Layer

@layer
def recurrent_layer(inputs, cell, name = None, initializer = None, sequence_length = None, initial_state = None, dtype = tf.float32, cell_bw = None):
    if initializer is None:
        initializer = tf.orthogonal_initializer()
    with tf.variable_scope(name or 'recurrent_layer', initializer = initializer):
        if cell_bw is None:
            outputs, state = rnn.dynamic_rnn(cell, inputs, sequence_length = sequence_length, initial_state = initial_state, dtype = dtype, swap_memory = True)
        else:
            outputs, state = rnn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, sequence_length = sequence_length, initial_state_fw = initial_state, dtype = dtype, swap_memory = True)
        return outputs, state
@layer
def embedding_layer(inputs, size, dim,  name = None, embedding=None, initializer=None, trainable = True):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    if name is None: name = "embedding"

    if embedding is None:
        embed_var = tf.get_variable(name + '_W', [size, dim], tf.float32, initializer = initializer, trainable = trainable)
    else:
        embed_var = tf.get_variable(name + '_W', initializer = embedding, trainable = trainable)
    embeded = tf.nn.embedding_lookup(embed_var, inputs, name = name)
    return embeded
def check_flatten(inputs, rank_limit):
    shape = inputs.get_shape().as_list()
    if len(shape) > rank_limit:
        if rank_limit == 1:
            inputs = tf.reshape(inputs, [-1])
        else:
            inputs = tf.reshape(inputs, [-1, shape[-1]])
    return inputs

@layer
def dense_layer(inputs, dim, name = None, initializer = None, input_dim = None, sparse = False, trainable = True, is_training=True, batch_norm = False, bias = 0.1, act=None):
    if name is None: name = "dense"

    shape = inputs.get_shape().as_list()
    inputs = check_flatten(inputs, 2)
    if input_dim is None:
        input_dim = shape[-1]

    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)

    w = tf.get_variable(name + "_w", [input_dim, dim], tf.float32, trainable = trainable, initializer = initializer)
    b = tf.get_variable(name + "_b", None, tf.float32, trainable = trainable, initializer = tf.ones(shape = dim)*bias)

    if sparse:
        out = tf.add(tf.sparse_tensor_dense_matmul(inputs, w), b, name = name)
    else:
        out = tf.add(tf.matmul(inputs, w), b, name = name)
    if batch_norm:
        print("%%%:batch norm enabled")
        out = tf.layers.batch_normalization(out, training = is_training)
    if act is not None:
        out = act(out)
    return out



class RNN(Model):
    _enc_dim = [128]
    _emb_dim = 64
    _layers = [128]
    _max_len = 100
    _use_w2v = True
    _emb_fname = 'w2v.csv'
    #_lr_decay_rate = 0.999
    #_epochs = 2
    #_lr = 1e-3
    _lr = 3e-3
    #_lr = 1e-2
    _es = 0.001
    _min_cnt = 5
    _train_embedding = True
    _dropout = None
    _tokenizer = 'tok_jieba'
    _decay_epoch = False
    def __init__(self, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._create_session()
        if args.char and self._emb_fname:
            self._emb_fname = 'char_' + self._emb_fname
        elif args.tok and self._emb_fname:
            self._emb_fname = args.tok + '_' + self._emb_fname

    def _create_session(self):
        self._sess = tf.Session()
        tf.set_random_seed(self.seed)


    def fit(self, x, y, validation_data = None, save = True, restore = True):
        if args.repredict:
            self.restore()
            pred = self.predict(validation_data[0])
            self._save_predict(pred, validation_data)
            return
        #self.xT = x; self.yT = y,; self.xV = validation_data[0]; self.yV = validation_data[1]
        self._bincnt = np.bincount(y)
        if self._model is None:
            if isinstance(x, dict):
                dim = None
            else:
                dim = x.shape[1]
            self._create_model(dim)
        if args.restore and restore:
            self.restore()
        with self._sess.graph.as_default():
            best_val_loss = np.inf
            if self._epochs is None:
                self._epochs = args.epochs
                no_val = False
            else:
                no_val = True
            if args.debug:
                self._epochs = min(self._epochs,10)
            for i in range(self._epochs):
                loss, val_loss= self._fit_epoch(x, y, validation_data, no_val, epoch=i)
                if not loss <100:
                    logging.error('something is wrong loss is:%s, exit', loss)
                    raise Exception('loss:{} is abnormal'.format(loss))
                #if not no_val:
                if 1==1:
                    #score = self.score(validation_data[0], validation_data[1], self._pred_thr)
                    #logging.info("epoch %s done, score is: %s", i, score)
                    if (best_val_loss-val_loss) / val_loss < -self._es:
                        logging.info("train done")
                        if not no_val:
                            break
                    else:
                        if best_val_loss > val_loss:
                            best_val_loss = val_loss
                    #self._batch_size = min(int(self._batch_size*1.2), self._max_batch_size)
                else:
                    score=-99; val_loss = -99
                if i<(self._epochs-1) and (time.time()-g_start_time)>g_allow_time and save:
                    logging.info('time limit exceed, will save model and return')
                    self.save()
                    logging.info('time limit exceed, model saved')
                    return
                if save:
                    self.save()
        if save:
            self.restore()
        pred = self.predict(validation_data[0])
        if save:
            self._save_predict(pred, validation_data)
        pred = pred > self._pred_thr
        score = self.f1_score(validation_data[1], pred)
        g_result[self.name] = (i,"{0:.4f}".format(loss),"{0:.4f}".format(val_loss), "{0:.4f}".format(score))

        if args.dump_embeddings:
            embddings = self._sess.run( self._sess.graph.get_tensor_by_name('embedding_layer/embedding_W:0'))
            with open(os.path.join('data', 'RNN_trained_embdddings.dump'), 'wb') as f:
                pickle.dump(embeddings, f)
    def _create_cell(self):
        cells = []
        for layer in self._enc_dim:
            cells.append(rnn_cell.GRUCell(layer))
        mult_lstm = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        return mult_lstm

    def _add_enc(self, s, s_seqs_len, emb, bi = False):
        self._s_emb = emb(s)
        if self._dropout:
            self._s_emb = tf.nn.dropout(self._s_emb, 1-self._dropout)
            
        outs, enc = self._rnn(self._s_emb)
        enc = enc[0]
        return enc, outs


    def _save_var(self):    
        fpath = self._gen_fname('', 'vars.dump')
        with open(fpath, 'wb') as f:
            pickle.dump((self._word_cnt, self._word_index, self._word_freq), f)
    def _restore_var(self):
        fpath = self._gen_fname('', 'vars.dump')
        with open(fpath,'rb') as f:
            self._word_cnt, self._word_index, self._word_freq = pickle.load(f)
            self._embeddings = None
    def save(self, global_step = None):
        super(MLP, self).save()
        fpath = self._gen_fname('', 'model')
        with self._sess.graph.as_default():
            saver = tf.train.Saver()
        save_path = saver.save(self._sess, fpath, global_step = global_step)
        logging.info("Model saved to file:{}".format(save_path))


    def restore(self, global_step = None):
        self._restore_var()
        if self._model is None:
            self._create_model(None)

        with self._sess.graph.as_default():
            saver = tf.train.Saver()
            fpath = self._gen_fname('')
            if global_step is None:
                ckpt = tf.train.get_checkpoint_state(fpath)
                model_path = ckpt.model_checkpoint_path
            else:
                model_path = fpath + "-{}".format(global_step)
            global_step = None
            saver.restore(self._sess, model_path)
            logging.info("Model restored from file:{}".format(model_path))

        return global_step


        
    def _add_train_op(self, loss, var_list = None):
        self._global_step = tf.Variable(0, trainable=False, name = 'global_step')
        lr_node = tf.train.exponential_decay(learning_rate=self._lr_plh, global_step=self._global_step, staircase=False, decay_steps=1, decay_rate=self._lr_decay_rate, name = 'learning_rate')
        self._lr_node = lr_node
        opt = tf.train.AdamOptimizer(learning_rate=lr_node)

        #max_grad = 1000000
        if var_list is None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._var_list = var_list
        gradients = opt.compute_gradients(loss, var_list)
        self._grads = gradients
        train_op = opt.apply_gradients(self._grads, global_step=self._global_step, name = 'train_op')
        return train_op


    def get_feed(self, batch):
        feed = {
            self._s1: batch['s1'],
            self._s2: batch['s2'],
            self._s1_seqs_len: batch['s1_seqs_len'],
            self._s2_seqs_len: batch['s2_seqs_len'],
            self._lr_plh: self._lr
            }
        if 'y' in batch:
            y = batch['y']
            feed[self._y] = y
            feed[self._dropout_plh] = self._dropout
        else:
            feed[self._dropout_plh] = 0.0

        return feed
    def _add_summary(self):
        tf.summary.scalar(self.loss.name, self.loss)
        tf.summary.scalar(self._lr_node.name, self._lr_node)
        for node in self._var_list:
            tf.summary.histogram(node.name, node)
        self._summary_writer = tf.summary.FileWriter(os.path.join('data', self.name), graph=self._sess.graph)
        self._summary_op = tf.summary.merge_all()
    def _create_input(self):
        self._y = tf.placeholder(tf.float32, [None], name = "y")
        self._dropout_plh = tf.placeholder(tf.float32, name = "dropout_plh")
        self._lr_plh = tf.placeholder(tf.float32, name = "lr_plh")
        self._s1 =  tf.placeholder(tf.int32, [None, None], name = "s1")
        self._s1_seqs_len = tf.placeholder(tf.int32,[None], name = 's1_seqs_len')
        self._s2 =  tf.placeholder(tf.int32, [None, None], name = "s2")
        self._s2_seqs_len = tf.placeholder(tf.int32,[None], name = 's2_seqs_len')
        num_s1 = self._s1.get_shape().as_list()[0]
        self._num_s1 = num_s1
        s = tf.concat([self._s1,self._s2], 0)
        s_seqs_len = tf.concat([self._s1_seqs_len,self._s2_seqs_len], 0)
        self._s = s; self._s_seqs_len = s_seqs_len
    def _create_enc(self, emb, bi = False):
        s_seqs_len = self._s_seqs_len; s = self._s
        self._cell = self._create_cell()
        if bi:
            self._cell_bw = self._create_cell()
            self._rnn = recurrent_layer(self._cell, sequence_length = s_seqs_len, cell_bw = self._cell_bw)
        else:
            self._rnn = recurrent_layer(self._cell, sequence_length = s_seqs_len)

        enc, hs = self._add_enc(s, s_seqs_len, emb, bi)
        self._hs = hs
        return enc
    def _add_comb(self):
        enc = self._enc
        enc1, enc2 = tf.split(enc, 2, 0)
        #hs1, hs2 = tf.split(enc, 2, 0)
        #hs1, hs2 = tf.split(hs, 2, 0)
        #self._hs1 = hs1; self._hs2 = hs2
        #enc1 = self._atten(enc2, hs1, self._s1_seqs_len)
        #enc2 = self._atten(enc1, hs2, self._s2_seqs_len)
        self._enc1 = enc1; self._enc2 = enc2


        #comb = tf.concat([tf.abs(enc1-enc2),enc1*enc2], -1)
        comb = tf.concat([tf.abs(tf.nn.l2_normalize(enc1,-1)-tf.nn.l2_normalize(enc2,-1)),tf.nn.l2_normalize(enc1*enc2, -1)], -1)
        #comb = tf.abs(tf.nn.l2_normalize(enc1,-1)-tf.nn.l2_normalize(enc2,-1))
        #comb = tf.concat([enc1, enc2], -1)
        self._comb = comb
        logging.info("concated shape %s", comb.shape)
    def _create_pred(self):
        self._add_comb() 
        fuse = self._comb
        for layer in self._layers:
            fuse = dense_layer(layer, act = tf.nn.relu)(fuse)
        out = tf.squeeze(dense_layer(1, act = None)(fuse), -1)
        #out = 1- tf.squeeze(tf.losses.cosine_distance(tf.nn.l2_normalize(enc1, -1),tf.nn.l2_normalize(enc2,-1), reduction= tf.losses.Reduction.NONE , axis = -1),-1)/2
        #out = tf.nn.relu(out)
        self._pred = tf.sigmoid(out)
        #self._pred = out
        #_epsilon = 1e-7
        #out = tf.clip_by_value(out, _epsilon, 1 - _epsilon)
        #out = tf.log(out / (1 - out))
        self._out = out
    def _create_model(self, inShape):
        self.loss_nodes = []
        with self._sess.graph.as_default():
            self._create_input()
         
            emb = embedding_layer(self._word_cnt, self._emb_dim, trainable=True, embedding = self._embeddings)
            #self._emb = embedding_layer(self._word_cnt, self._emb_dim, trainable=True)
            self._enc = self._create_enc(emb)
            self._create_pred()
            loss = self._add_loss()
            self.loss_nodes.append(loss)
            self.loss = tf.add_n(self.loss_nodes)
            #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._out), name='loss_sim')
            self._train_op = self._add_train_op(self.loss)
            self._var_init = tf.global_variables_initializer()
            #self._add_summary()
            self._model = self
            self._sess.run(self._var_init)
    def _add_loss(self):
        #dist = tf.squeeze(tf.losses.cosine_distance(tf.nn.l2_normalize(self._enc1, -1),tf.nn.l2_normalize(self._enc2,-1), reduction= tf.losses.Reduction.NONE , axis = -1),-1)
        #self.loss_dist = tf.reduce_mean(1*(self._y*dist + (1-self._y) * tf.maximum(1.5-dist, 0)), name='loss_dist')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._out), name='loss_sim')
        #loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self._y, logits=self._out, pos_weight=self._bincnt[0]/self._bincnt[1]), name='loss_sim')
        #self.loss_nodes = [self.loss_dist, self.loss_sim ]
        #self.loss = tf.add_n(self.loss_nodes)
        return loss
    def _get_batch(self, x, y = None, shuffle = True):
        num = x['s1'].shape[0]
        inds = np.arange(num)
        if shuffle:
            self._rs.shuffle(inds)
        num_batch = (num + self._batch_size -1)//self._batch_size
        for i in range(num_batch):
            batch = {}
            batch_inds = inds[self._batch_size*i:min(self._batch_size*(i+1), num)]
            batch['batch_inds'] = batch_inds
            for ks, kl in [('s1','s1_seqs_len'), ('s2', 's2_seqs_len')]:
                batch[kl] = x[kl][batch_inds]
                batch[ks] = x[ks][batch_inds, :]
            max_len = max(np.max(batch['s1_seqs_len']), np.max(batch['s2_seqs_len']))
            #max_len = max(np.max(x['s1']), np.max(x['s2']))
            for ks in ['s1','s2']:
                batch[ks] = batch[ks][:, :max_len]
            randv = np.random.rand()
            if randv>0.5:
                for k1, k2 in [('s1','s2'), ('s1_seqs_len', 's2_seqs_len')]:
                    tmp = batch[k1]
                    batch[k1] = batch[k2]
                    batch[k2] = tmp
                        
            if y is not None:
                batch['y'] = y[batch_inds]
#                batch['y'] = np.concatenate([batch['y'], np.zeros([len(batch['y'])],np.int32)],0)
#                batch['s1'] = np.concatenate([batch['s1'], batch['s1']],0)
#                batch['s2'] = np.concatenate([batch['s2'], batch['s2'][::-1,::]],0)
#                batch['s1_seqs_len'] = np.concatenate([batch['s1_seqs_len'], batch['s1_seqs_len']],0)
#                batch['s2_seqs_len'] = np.concatenate([batch['s2_seqs_len'], batch['s2_seqs_len']],0)
            yield batch

    def run(self, sess, batch, nodes):
        feed = self.get_feed(batch)
        outputs = sess.run(nodes, feed)
        return outputs
    def _fit_epoch(self, x, y, validation_data = None, no_val=False, epoch=None):
        if epoch is not None and self._decay_epoch:
            self._lr = self._lr/(epoch+1)
        if validation_data is not None:
            xV, yV = validation_data
        losses = []; outstr = ""; val_losses = []
        for node in self.loss_nodes:
            losses.append([])
            val_losses.append([])
            outstr += node.name + ":{},"
            
        for i, batch in enumerate(self._get_batch(x, y)):
            outs = self.run(self._sess, batch, self.loss_nodes + [self._global_step, self._train_op])
            #outs = self.run(self._sess, batch, [self.loss, self._global_step, self._summary_op, self._train_op])
            for j, node in enumerate(self.loss_nodes):
                losses[j].append(outs[j])
            global_step = outs[j+1]; summary = outs[j+2]
            #self._summary_writer.add_summary(summary, global_step)
            #self._summary_writer.flush()
            #sys.stdout.write(('\r' + outstr).format(*map(np.mean, losses)))
            #sys.stdout.flush()
        logging.info('name:%s,global step:%s,train loss is:%s, totally %s batchs', self.name,global_step, outstr.format(*map(np.mean, losses)), i)
        loss = np.sum(list(map(np.mean, losses)))
        #if not no_val:
        if 1==1:
            for i, batch in enumerate(self._get_batch(xV, yV)):
                outs = self.run(self._sess, batch, self.loss_nodes + [self._global_step])
                for j, node in enumerate(self.loss_nodes):
                    val_losses[j].append(outs[j])
                global_step = outs[j+1]
            val_loss = np.sum(list(map(np.mean, val_losses)))
            logging.info(("epoch:{}, val loss is " + outstr).format(epoch, *map(np.mean, val_losses)))
        else:
            val_loss = None
        #pred = self.predict(x)
        #logging.info("train score is %s", self.score(x,y))
        return loss, val_loss


    def predict(self, x):    
        preds = []
        for i, batch in enumerate(self._get_batch(x, shuffle=False)):
            pred = self.run(self._sess, batch, self._pred)
            preds.append(pred)
        return np.concatenate(preds,0)
#    def _load_embeddings(self, word_index):
#        fname = os.path.join('data', self.name + '_embeddings.dump')
#        if not os.path.exists(fname):
#            embeddings = load_embeddings(word_index, self._embedding_fpath)
#            with open(fname, 'wb') as f:
#                pickle.dump(embeddings, f)
#        else:
#            logging.info('load from existed:%s', fname)
#            with open(fname, 'rb') as f:
#                embeddings = pickle.load(f)
#        return embeddings
    def format_data(self, df, pctV = 0.05, dT = None, dV = None):
        
        dT, dV = super(RNN, self).format_data(df, pctV, dT, dV)
        if dT is not None:
            self._word_index = None
            self._word_cnt = None
            self._word_freq = None
            if not args.emb_fname:
                dirname = self._gen_fname('')
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                w2v(seed=self.seed, emb_dim = self._emb_dim, data_dir = dirname, df = dT, tokenizer = gl[self._tokenizer], vp = pctV, min_cnt = self._min_cnt)
        if args.emb_fname:
            emb_dir = data_dir
        else:
            emb_dir = self._gen_fname('')
        xT, xV, yT, yV, dT, dV, word_index, embeddings, word_freq = seq(dT, dV, self._max_len, self._word_index, self._min_cnt, self._emb_fname, data_dir = emb_dir, tokenizer = gl[self._tokenizer])
        if word_index is not None:
            self._embeddings = embeddings
            self._word_cnt = len(word_index)
            self._word_index = word_index
            self._word_freq = word_freq

        return xT, xV, yT, yV, dT, dV
class RNNEM128(RNN):
    _emb_dim = 128

class ELMO(RNN):
    _epochs = None
    _sim_epochs = None
    _lm_epochs = None
    #_lr_decay_rate = 0.9999
    _lr_decay_rate_lm = 1.0
    _lr_decay_rate_sim = 1.0
    _lm_batch_size=200
    _batch_size = 500
    #_emb_dim = 128
    _enc_lm_dim = [128]
    _enc_sim_dim = [128]
    _lr_lm = 3e-3
    _lr_sim = 3e-3
    _lr_s2 = 5e-6
    _shar_map = False
    _2stage = False
    _stage = 0
    def _add_s_emb(self):
        with tf.variable_scope("elmo_att", reuse=tf.AUTO_REUSE):
            #anchor = tf.get_variable("elmo_att", [1,1,2], tf.float32, initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            self._elmo_anchor = tf.get_variable("elmo_att",  shape=None,  dtype=tf.float32, initializer =  np.ones([1,1,1,2],np.float32))
            self._elmo_scale = tf.get_variable("elmo_scale", shape=None, dtype = tf.float32, initializer =  1.0)

        att = tf.nn.softmax(self._elmo_anchor) + 1e-14
        comb = tf.concat([tf.expand_dims(self._s_emb, -1), tf.expand_dims(self._hs,-1)], -1)
        s_emb = self._elmo_scale*tf.reduce_sum(comb*att, -1)
        return s_emb
    def _add_enc(self, s, s_seqs_len, emb = None, bi = False):
        if emb is None:
            self._s_emb = self._add_s_emb()
        else:
            self._s_emb = emb(s)
        if self._dropout and emb is None:
            logging.info('dropout:%s', self._dropout)
            self._s_emb = tf.nn.dropout(self._s_emb, 1-self._dropout_plh)

        outs, enc = self._rnn(self._s_emb)
        #outs = (outs[0]+outs[1])/2
        #outs = tf.concat(outs, -1)
        #enc = enc[0]
        #enc = tf.reduce_mean(emb, 1)
       
        #if bi:
        #    outs1,outs2= outs
        #    enc1 = tf.gather_nd(outs1, tf.stack([tf.range(tf.shape(outs1)[0]), s_seqs_len -1], axis = 1))
        #    enc2 = tf.gather_nd(outs2, tf.stack([tf.range(tf.shape(outs2)[0]), s_seqs_len -1], axis = 1))
        #    enc = (enc1, enc2)
        #else:
        #    enc = tf.gather_nd(outs, tf.stack([tf.range(tf.shape(outs)[0]), s_seqs_len -1], axis = 1))
        if bi:
            enc = enc
            
        else:
            enc = enc[0]
        return enc, outs
    def _add_lm(self, hs, seqs, seqs_len):
        shape = tf.shape(hs[0])
        weights = tf.cast(tf.sequence_mask(seqs_len-1, shape[1] -1), tf.float32)
        h1, h2 = hs
        self._map = dense_layer(self._emb_dim//2)
        h1 = h1 >> self._map
        h1 = tf.reshape(h1, [shape[0], shape[1], self._emb_dim//2])
        if not self._shar_map:
            self._map = dense_layer(self._emb_dim/2)
        h2 = h2 >> self._map
        h2 = tf.reshape(h2, [shape[0], shape[1], self._emb_dim//2])
        self._hs = tf.concat([h1,h2],-1)

        self._projecting = dense_layer(self._word_cnt)
        logits = h1[:, :-1,:]  >> self._projecting
        logits = tf.reshape(logits, [shape[0], shape[1]-1, self._word_cnt])
        self._lm_loss_1 = tf.reduce_mean(weights*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=seqs[:,1:]), name='lm_loss_1')

        #self._projecting = dense_layer(self._word_cnt)
        logits = h2[:, 1:,:] >> self._projecting
        logits = tf.reshape(logits, [shape[0], shape[1]-1, self._word_cnt])
        self._lm_loss_2 = tf.reduce_mean(weights*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=seqs[:,:-1]), name='lm_loss_2')

        self._var_list_lm = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self._lr = self._lr_lm
        self._lr_decay_rate = self._lr_decay_rate_lm
        self._train_op_lm = self._add_train_op(self._lm_loss_1 + self._lm_loss_2)

    def _save_var(self):    
        fpath = self._gen_fname('', 'vars.dump')
        with open(fpath, 'wb') as f:
            pickle.dump((self._stage, self._word_cnt, self._word_index, self._word_freq), f)
    def _restore_var(self):
        fpath = self._gen_fname('', 'vars.dump')
        with open(fpath,'rb') as f:
            self._embeddings = None
            var = pickle.load(f)
        if len(var) == 4:
            self._stage, self._word_cnt, self._word_index, self._word_freq = var
        else:
            self._word_cnt, self._word_index, self._word_freq = var

    def fit(self, x, y, validation_data = None, save = True):
        if self._model is None:
            self._create_model(None)
        if args.restore:
            self._restore_var()

        self._lr = self._lr_lm
        self._es_bak = self._es
        self._es = 0.001
        self._train_op = self._train_op_lm
        self.loss_nodes = [self._lm_loss_1, self._lm_loss_2]
        self._batch_size_bak = self._batch_size
        self._batch_size = self._lm_batch_size
        self._epochs = self._lm_epochs
        if self._stage == 0:
            logging.info('***** fit for LM, lr:%s', self._lr)
            super(ELMO, self).fit(x, y, validation_data, False)
            elmo_scale, elmo_anchor = self._sess.run([self._elmo_scale, self._elmo_anchor])
            logging.info('***** end fit for LM, scale:%s, anchor:%s', elmo_scale, elmo_anchor)
            self._stage = 1
            self._save_var()

        self.loss_nodes = [self._loss_sim]
        self._es = self._es_bak
        self._batch_size = self._batch_size_bak
        self._epochs = self._sim_epochs
        self._lr = self._lr_sim
        if self._stage ==1: 
            if self._2stage:
                logging.info('***** sim stage1, lr:%s', self._lr)
                self._train_op = self._train_op_sim1
                super(ELMO, self).fit(x, y, validation_data, save)
            self._stage =2
            self._save_var()
        if self._stage ==2:
            if self._2stage:
                self._lr = self._lr_s2
            logging.info('***** sim stage2, lr:%s', self._lr)
            self._epochs = self._sim_epochs
            self._train_op = self._train_op_sim2
            super(ELMO, self).fit(x, y, validation_data, save, restore=True)
            elmo_scale, elmo_anchor = self._sess.run([self._elmo_scale, self._elmo_anchor])
            logging.info('***** end fit, scale:%s, anchor:%s', elmo_scale, elmo_anchor)


    def _create_model(self, inShape):
        #super(LMRNN, self)._create_model(inShape)
        with self._sess.graph.as_default():
            self.loss_nodes = []
            self._create_input()

            ##for lm
            self._enc_dim = self._enc_lm_dim
            emb = embedding_layer(self._word_cnt, self._emb_dim, trainable=True, embedding = self._embeddings)
            #emb = embedding_layer(self._word_cnt, self._emb_dim, trainable=True, embedding = None)
            self._enc = self._create_enc(emb, bi=True)
            self._add_lm(self._hs, self._s, self._s_seqs_len)
            self.loss_nodes = [self._lm_loss_1, self._lm_loss_2]

            ##for sim
            self._enc_dim = self._enc_sim_dim
            self._enc = self._create_enc(None)
            #self._pred = self._create_pred()
            self._create_pred()
            self._loss_sim = self._add_loss()
            self.loss_nodes = [self._loss_sim]
            #self.loss = tf.add_n(self.loss_nodes)
            #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._y, logits=self._out), name='loss_sim')
            self._var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._lr = self._lr_sim
            if self._2stage:
                self._var_list_sim = [var for var in self._var_list if var not in self._var_list_lm]
                self._train_op_sim1 = self._add_train_op(self._loss_sim, var_list = self._var_list_sim)
                self._lr = self._lr_s2

            self._lr_decay_rate = self._lr_decay_rate_sim
            self._train_op_sim2 = self._add_train_op(self._loss_sim, var_list = self._var_list)


            self._var_init = tf.global_variables_initializer()
            self._sess.run(self._var_init)
        self._model = self

class ELMOEM128(ELMO):
    _emb_dim=128



def load_preds(names):
    preds = []
    dV = load_validate()
    for name in names:
        model = create_model(name)
        fname = model._gen_fname('', 'pred.csv')
        pred = pd.read_csv(fname)
        preds.append(pred.pred.values)
    return preds, dV
def cal_best_ens(names = None, preds = None, dV = None, per_m = True):
    if names is None:
        names = parse_model_name(args.model_names)
        preds, dV = load_preds(names)
        
    avg_pred = np.mean(preds, 0)
    avg_pred_thr, avg_score = search_pred_thr(avg_pred, dV.y.values)
    
    preds_dict = defaultdict(list)
    for name, pred in zip(names, preds):
        name = '_'.join(name.split('_')[0:-1])
        preds_dict[name].append(pred)
    if per_m:
        for name, pred in preds_dict.items():
            preds_dict[name] = np.mean(preds_dict[name], 0)
            pred_thr, score = search_pred_thr(preds_dict[name], dV.y.values)
            logging.info('      avg for model:%s, score:%s, pred_thr:%s', name, score, pred_thr)
    #preds_dict = dict(zip(names, preds))
    best_score = -1; best_names = []; best_preds = []; best_pred_thr = None
    while len(preds_dict)>0:
        find_best = False
        for name, pred in preds_dict.items():
            pred_tmp = np.mean(best_preds + [pred], 0)
            pred_thr, score = search_pred_thr(pred_tmp, dV.y.values)
            if score > best_score:
                best_score = score
                best_name = name
                best_pred = pred
                best_pred_thr = pred_thr
                find_best = True
        if find_best:
            logging.info('  find best model:%s, score:%s, pred_thr:%s', best_name, best_score, best_pred_thr)
            best_preds.append(best_pred)
            best_names.append(best_name)
            preds_dict.pop(best_name)
        else:
            break
    logging.info('avg score:%s,prd_thr:%s', avg_score,avg_pred_thr)
    logging.info('best ens score:%s, pred_thr:%s, models:%s', best_score, best_pred_thr, ','.join(best_names))
    return best_names, best_preds


            
    
def parse_model_name(model_names):
    return model_names.replace(' ','').split(',')
    
def kf(k=10, save=None):
    start_t =time.time()
    #if re.search('T2S', args.model_names.upper()):
    #    args.t2s = True
    if save is None:
        save = args.save
    names = parse_model_name(args.model_names)
    df =load_data()
    if args.debug:
        df = df[0:g_debug_cnt]
    kf = KFold(n_splits=k, random_state=np.random.RandomState(8763))
    splits = kf.split(df)
    kf_result = {}
    for name_id, name in enumerate(names):
        preds = []; ys = []; predTs = []
        ex_feas = None
        for i in range(k):
            train_ids, valid_ids = next(splits)
            dT = df.iloc[train_ids]; dV = df.iloc[valid_ids]
            if args.kfid is not None:
                if str(i) not in  args.kfid.split(' '):
                    continue
            logging.info("*************888id:%s", i)
            model_name = name + '_KF'+str(i)
            model = create_model(model_name, name_id)
            if isinstance(model,ELMOEX):
                if ex_feas is None:
                    use_char=args.char
                    ex_feas = load_ex(model._ex_kf, model._ex_names, df=None)
                    if 'mis_feas' in df.columns and model._use_mis:
                        mis_feas = np.array(list(df['mis_feas'].values))
                        ex_feas = np.concatenate([ex_feas,mis_feas],-1)
                    args.char = use_char
                dT['ex_feas'] = list(ex_feas[train_ids])
                dV['ex_feas'] = list(ex_feas[valid_ids])
            #model.data_dir = os.path.join(data_dir, 'kf')
            #if isinstance(model, RNN):
            #    w2v(seed=model.seed, emb_dim = model._emb_dim, data_dir = model._gen_fname(''), df = dT, tokenizer = gl[model._tokenizer], min_cnt = model._min_cnt, workers = 1)
            xT, xV, yT, yV, dT, dV  = model.format_data(None, dT = dT, dV = dV)
            model.fit(xT, yT, (xV, yV), args.save)
            logging.info('train done for modle:%s,start predict', model_name)
            pred = model.predict(xV)
            preds.append(pred)
            ys.append(dV.y.values)

            pred_tmp = np.concatenate(preds)
            pred_thr, score = search_pred_thr(pred_tmp, np.concatenate(ys))
            kf_result[model_name] = (i, pred_thr, score)
            logging.info('******score is:%s, pred_thr:%s, kf:%s', score, pred_thr, i)

            del model
    logging.info('***** kf result:%s, time:%s', kf_result, time.time()-start_t)
        
def create_model(name, seed=None):
    name = name.upper()
    if re.search('CHAR', name):
        args.char=True
    else:
        args.char=False
    cls = name.split('_')[0]
    if args.suffix is not None:
        name = name + '_' + args.suffix
    #if args.cvid:
    #    name = name + '_cv'+str(args.cvid)

    model = gl[cls](name = name)
    if seed:
        model.seed = seed
        model._rs = np.random.RandomState(model.seed)
    if args.debug:
        model._batch_size = 30
    if args.es:
        model._es = args.es
    if args.lr:
        model._lr = args.lr
    if args.emb_dim:
        model._emb_dim = args.emb_dim
    if args.mc is not None:
        model._min_cnt= args.mc
    if args.emb_fname:
        model._emb_fname = args.emb_fname
    if args.dropout:
        model._dropout = args.dropout

    if model._emb_fname:
        model._emb_fname = str(model._emb_dim)+'d_' + model._emb_fname
    logging.info('model:%s', name)
    return model
    
    
def train_model(dT = None, dV = None, model_names = None, vp = None):
    if model_names is None:
        model_names = args.model_names
    #if re.search('T2S', model_names.upper()):
    #    args.t2s = True
    if dT is None:
        df =load_data()
    else:
        df = None
    if vp is None:
        vp = args.vp
    if args.debug:
        df = df[0:g_debug_cnt]
    preds = []
    names = parse_model_name(model_names)
    #if args.t2s:
    #    for name in names:
    #        assert re.search('T2S', name.upper()), 'all name must have t2s'
        
    result = {}
    for i, name in enumerate(names):
        logging.info('will train for model: %s', name)
        model = create_model(name, seed=int(time.time()))
        xT, xV, yT, yV, dT, dV  = model.format_data(df, vp, dT = dT, dV = dV)
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
        model.fit(xT, yT, (xV, yV), args.save)
        logging.info('train done for modle:%s,start predict', name)
        pred = model.predict(xV)
        #pred_thr, score  =  search_pred_thr(pred, dV = dV):
        #logging.info('  best_predh: %s', pred_thr)
        preds.append(pred)
        logging.info('  pred mean: %s', np.mean(pred))
        del model
        tf.reset_default_graph()
    best_names, best_preds = cal_best_ens(names, preds, dV, True)
    logging.info('resut is :%s', g_result)
    
    return best_names, best_preds

def pred_model(name, df):
    model = create_model(name)
    model.restore()
    xT, xV, yT, yV, dT, dV  = model.format_data(df, pctV = None)
    if model.name.endswith('B'):
        xV = xV.astype(np.bool).astype(np.float32)
    if isinstance(model, PAIR):
        pred = model.predict(xV, avg=False)
    else:
        pred = model.predict(xV)
    logging.info('pred mean:%s', np.mean(pred))
    model._sess.close()
    del model
    tf.reset_default_graph()
    return pred
def load_validate(fname = 'validate.csv', vp = None):    
    if vp is None:
        vp = args.vp
    fname = 'vp'+str(int(args.vp*100)) + '_' + fname
    if args.debug:
        fname = 'debug_' + fname
    dV = pd.read_csv(os.path.join(data_dir, fname))
    return dV
def save_validate(dV, fname = 'validate.csv', vp = None):
    if vp is None:
        vp = args.vp
    fname = 'vp'+str(int(args.vp*100)) + '_' + fname
    if args.debug:
        fname = 'debug_' + fname
    dV.to_csv(os.path.join(data_dir, fname), index=False)
def eval_model(use_pair=False):
    single = args.single
    combine = args.combine
    #if re.search('T2S', args.model_names.upper()):
    #    args.t2s = True
    fpath = os.path.join(data_dir,'eval_model')
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    model = Model()
    df = load_data(use_local=True)
    if use_pair:
        ex_feas = load_ex(10, ['PAIR'])
        df['ex_feas'] = list(ex_feas)
    _, dV = train_test_split(df, test_size = args.vp, random_state=np.random.RandomState(8763))
    if g_local:
        df = dV.copy()
    else:
        df = load_data(random=False)
        ex_feas = load_ex(10, ['PAIR'], df)
        df['ex_feas'] = list(ex_feas)
    #if args.debug:
    #    df = df[0:100]
    df['y'] = -1; preds = []; predsV = []
    #for name in ['MLP_TFIDF1000','MLP_TFIDF1000B', 'MLP_CV1500', 'MLP_CV1500B', 'RNN']:
    #for name in ['MLP_TFIDF1000B', 'MLP_CV1500', 'MLP_CV1500B', 'RNN']:
    #for name in ['MLP_TFIDF1000B', 'MLP_CV1500B', 'RNN']:
    names = parse_model_name(args.model_names)
    model_names = []
    if combine:
        preds=[]; predsV = []
        for name in names:
            pred = pd.read_csv(os.path.join(fpath, name +'_pred.csv')).pred.values
            preds.append(pred)
        predsV,_ = load_preds(names)
    else:
        for name in names:
            #for i in range(21,22):
            if 1==2:
                model_name = name + '_'+str(i)
            else:
                model_name = name
            model_names.append(model_name)
            #pred = pred_model(model_name, df)
            with Pool(1) as p:
                pred = p.apply(pred_model,(model_name, df))
            #predV = pred_model(name, dV)
            
            preds.append(pred)
            #predsV.append(predV)
                
        predsV,_ = load_preds(model_names)
    #predsV = preds
    pred = np.mean(preds, 0)
    predV = np.mean(predsV, 0)
    thr, _ = search_pred_thr(predV)
    if single:
        df = pd.DataFrame(columns=['pred'], data=pred)
        df.to_csv(os.path.join(fpath, model_names[0]+'_pred.csv'), index=False)
        dV = pd.DataFrame(columns=['predV'])
        return
    if args.cvlgb:
        mis_feas = np.array(list(df['mis_feas'].values))
        mis_feasV = np.array(list(dV['mis_feas'].values))
        ex_feas = np.array(list(df['ex_feas'].values))
        ex_feasV = np.array(list(dV['ex_feas'].values))
        x = np.concatenate([np.array(preds).T, mis_feas, ex_feas], -1)
        xV = np.concatenate([np.array(predsV).T, mis_feasV, ex_feasV], -1)
        with open(os.path.join(data_dir, 'lgbm.dump'), 'rb') as f:
            #models = pickle.load(f)
            models, best_pred_thr, best_score = pickle.load(f)
            logging.info('load lgbm, best_pred_thr %s, best_score %s', best_pred_thr, best_score)
            lgbm_preds = []
            lgbm_predsV = []
            for model in models:
                pred = model.predict(x)
                lgbm_preds.append(pred)
                predV = model.predict(xV)
                lgbm_predsV.append(predV)
            pred = np.mean(lgbm_preds, 0)
            predV = np.mean(lgbm_predsV, 0)
        logging.info('******** for lgbm')
        score = f1_score(predV>0.5,dV.y.values)
        logging.info('lgbm use thr 0.5 score is %s', score)
        #thr, _ = search_pred_thr(predV)
        #thr = 0.66
        thr = best_pred_thr
    del df['s1']
    del df['s2']
    df['y'] = pred
    logging.info('test pred mean is %s:', np.mean(pred))
    #dV.to_csv(os.path.join(data_dir, 'test_pred_prob.csv'), index=False)
    del df['y']


    #dV['label'] = (pred >model._pred_thr).astype(np.int32)
    df['label'] = (pred >thr).astype(np.int32)
    for col in ['mis_feas', 'ex_feas']:
        if col in df.columns:
            del df[col]
    if not single:
        if g_local:
            df.to_csv(os.path.join(data_dir, 'test_pred.csv'), index=False)
        else:
            topai(1, df)



    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-d", "--debug", action = "store_true", help="debug")
    parser.add_argument("-save", "--save", action = "store_true", help="save")
    parser.add_argument("-single", "--single", action = "store_true", help="save")
    parser.add_argument("-combine", "--combine", action = "store_true", help="save")
    parser.add_argument("-repredict", "--repredict", action = "store_true", help="save")
    parser.add_argument("-cvlgb", "--cvlgb", action = "store_true", help="save")
    parser.add_argument("-restore", "--restore", action = "store_true", help="save")
    parser.add_argument("-m", "--model_name", help="model name")
    parser.add_argument("-ms", "--model_names", help="model names")
    parser.add_argument("-suf", "--suffix", help="suffex model name")
    parser.add_argument("-de", "--dump_embeddings", help="dump embeddings")
    parser.add_argument("-eval", "--eval", action = "store_true", help="eval")
    parser.add_argument("-in", "--input", default=None, help="input")
    parser.add_argument("-out", "--output", default=None, help="output")
    parser.add_argument("-es", "--es", type=float, help="early stop value")
    parser.add_argument("-epochs", "--epochs", type=int, default=100, help="early stop value")
    parser.add_argument("-ext", "--ext", action = "store_true", help="ext")
    parser.add_argument("-ed", "--emb_dim", type=int,default=None, help="early stop value")
    parser.add_argument("-emb", "--emb_fname", default=None, help="input")
    parser.add_argument("-tok", "--tok", default=None, help="tokenizer")
    parser.add_argument("-char", "--char", action = "store_true", help="char")
    parser.add_argument("-correct", "--correct", action = "store_true", help="correct")
    parser.add_argument("-t2s", "--t2s", action = "store_true", help="t2s")
    parser.add_argument("-lr", "--lr", type=float,default=None, help="early stop value")
    parser.add_argument("-dp", "--dropout", type=float,default=None, help="early stop value")
    parser.add_argument("-mc", "--mc", type=int,default=5, help="early stop value")
    parser.add_argument("-cvid", "--cvid", type=int,default=None, help="early stop value")
    parser.add_argument("-kfid", "--kfid", default=None, help="kf")
    parser.add_argument("-rp", "--run_copy", type=int,default=1, help="early stop value")
    parser.add_argument("-vp", "--vp", type=float,default=0.1, help="valid_percent")
    parser.add_argument("-nv", "--no_validate", action = "store_true", help="not do validate")
    global args
    args = parser.parse_args()

    if not g_local:
        args.model_name = ATEC_MODEL_NAME
        args.model_names = ATEC_MODELS_NAME
        args.save = True
        #args.debug = True
        args.debug = False

    #for key in logging.Logger.manager.loggerDict:
    #    print(key)
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
        #logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(threadName)s %(message)s')
        #logger.setLevel(logging.INFO)
    #if not g_local and os.path.exists(model_dir):
    #    logger.info('list model_dir:%s', glob.glob(os.path.join(model_dir, '*')))
    #if not g_local:
    logging.getLogger('gensim').setLevel(logging.ERROR)
    logging.getLogger('jieba').setLevel(logging.ERROR)

    logging.info('################')
    global g_start_time 
    global g_allow_time 
    g_start_time = time.time()
    if g_local:
        g_allow_time = 650
    else:
        g_allow_time = 6500
    if args.debug:
        g_allow_time = 60
        
    if args.correct:
        for w in g_new_words.split(' '):
            jieba.add_word(w)
        
        for words in g_error_words.split(','):
            fm, to  = words.split(':')
            g_trans[fm.strip()] = to.strip()

    if args.eval:
        if args.input or args.output is None:
            raise Exception('must specify input and output')
        eval_model(args.input, args.output)
    else:
#        if args.model_name=='ext':
#            ext()
#            
#        elif args.model_name=='w2v':
#            w2v(args.input, args.output)
#        elif args.model_name=='eval_model':
#            eval_model()
        if args.model_name in gl:
            if inspect.isfunction(gl[args.model_name]):
                gl[args.model_name]()
            else:
                main()
        else:
            logging.error('unknown model %s', args.model_name)
    #if not g_local:
    #    if os.path.exists(model_dir):
    #        logging.info('list model_dir:%s', glob.glob(os.path.join(model_dir, '*')))
    logging.info('args:%s', args)
    sys.stdout.flush()
