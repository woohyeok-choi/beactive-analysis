import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List, NamedTuple, Dict, Set
from itertools import product
from datetime import datetime, timedelta
import asyncio
from abc import ABC, abstractmethod
from .dbstream import DBStream
from time import perf_counter
from pywt import wavedec
import os
import pickle


class Feature(NamedTuple):
    timestamp: int
    name: str
    value: float
    categories: Optional[Union[Set[str], Set[bool]]]


class ProcessedData(NamedTuple):
    id: Optional[str]
    train: pd.DataFrame
    label: pd.DataFrame
    categories: List[Optional[Tuple[int, Union[Set[str], Set[bool]]]]]

    def dump(self, path: str):
        d = os.path.join(path, self.id)
        os.makedirs(d, exist_ok=True)

        self.train.to_pickle(os.path.join(d, 'train'))
        self.label.to_pickle(os.path.join(d, 'label'))

        with open(os.path.join(d, 'categories'), 'wb') as f:
            pickle.dump(self.categories, f)

    @classmethod
    def load(cls, path: str) -> 'ProcessedData':
        name = os.path.split(path)[-1]
        train = pd.read_pickle(os.path.join(path, 'train'))
        label = pd.read_pickle(os.path.join(path, 'label'))

        with open(os.path.join(path, 'categories'), 'rb') as f:
            categories = pickle.load(f)

        return ProcessedData(
            id=name,
            train=train,
            label=label,
            categories=categories
        )


APP_CATEGORIES = {"LIFESTYLE": ["Com.sktelecom.minit", "Kal.FlightInfo", "app.hybirds.skypeople", "cc.pacer.androidapp",
                                "cheehoon.ha.particulateforecaster", "coinone.co.kr.official", "com.CouponChart",
                                "com.F1.ShinSG", "com.TouchEn.mVaccine.webs", "com.aboutjsp.thedaybefore",
                                "com.agoda.mobile.consumer", "com.aia.vitality.kr", "com.airbnb.android",
                                "com.airvisual", "com.ak.android.akmall", "com.alibaba.aliexpresshd",
                                "com.amanefactory.totsukitoka", "com.aminlogic.NavigationCompass", "com.app.shd.pstock",
                                "com.application_4u.qrcode.barcode", "com.bcf.coinstep", "com.bientus.cirque.android",
                                "com.blockfolio.blockfolio", "com.btckorea.bithumb",
                                "com.buzzni.android.subapp.shoppingmoa", "com.cafe24.ec.plusandar01",
                                "com.caloriecoin.wallet", "com.cashwalk.cashwalk", "com.chanel.weather.forecast.accu",
                                "com.chbreeze.jikbang4a", "com.cj.twosome", "com.cjkoreaexpress", "com.cjoshppingphone",
                                "com.coffeebeankorea.android", "com.coupang.mobile", "com.croquis.zigzag",
                                "com.cultsotry.yanolja.nativeapp", "com.delta.mobile.android", "com.dencreak.spbook",
                                "com.diningcode", "com.dnt7.threeW", "com.dominicosoft.coinmaster",
                                "com.dunamu.exchange", "com.dunamu.stockplus", "com.dunamu.trading",
                                "com.easy.currency.extra.androary", "com.ebay.kr.auction", "com.ebay.kr.g9",
                                "com.ebay.kr.gmarket", "com.ediya.coupon", "com.elevenst", "com.elevenst.deals",
                                "com.espn.score_center", "com.fitstar.pt", "com.fridaynoons.playwings",
                                "com.global.tarotmaster", "com.google.android.apps.maps", "com.gsr.gs25",
                                "com.habb.thankq", "com.hanabank.ebk.channel.android.hananbank",
                                "com.hanaskcard.paycla", "com.hanaskcard.rocomo.potal", "com.hanbit.fitday",
                                "com.hanbit.rundayfree", "com.hanwhalife.mobilecenter", "com.happy.mobilepop",
                                "com.hcom.android", "com.himart.main", "com.hmallapp", "com.hnsmall",
                                "com.hogangnono.hogangnono", "com.homeplus.myhomeplus", "com.hotelscombined.mobile",
                                "com.hpapp", "com.hyundaicard.appcard", "com.hyundaioilbank.android",
                                "com.ibillstudio.thedaycouple", "com.ibk.onebankA", "com.idincu.ovey.android",
                                "com.inbody2014.inbody", "com.interpark.shop", "com.interpark.tour.mobile.main",
                                "com.jasonmg.market09", "com.kakao.i.connect", "com.kakaobank.channel",
                                "com.kayak.android", "com.kbankwith.smartbank", "com.kbcard.cxh.appcard",
                                "com.kbcard.kat.liivmate", "com.kbcard.kbkookmincard", "com.kbstar.kbbank",
                                "com.kbstar.minibank", "com.kbstar.smartotp", "com.kbstar.starpush",
                                "com.kebhana.hanapush", "com.korail.talk", "com.kr.hollyscoffee",
                                "com.kt.android.showtouch", "com.kt.gigagenie.mobile", "com.kt.ktauth",
                                "com.ktcs.whowho", "com.ktshow.cs", "com.lcacApp", "com.lge.lifetracker",
                                "com.lguplus.mobile.cs", "com.linkzen.app", "com.looket.soim",
                                "com.lotte.lottedutyfree", "com.lotte.lpay", "com.lottemart.lmscp",
                                "com.lottemembers.android", "com.lpoint.moalock", "com.lunatouch.eyefilter.pro",
                                "com.makeshop.powerapp.jinsence81", "com.makeshop.powerapp.xexymix",
                                "com.mapmyrun.android2", "com.mo2o.alsa", "com.moms.momsdiary", "com.mozzet.lookpin",
                                "com.mrpizza.android", "com.mrt.ducati", "com.mysmilepay.app",
                                "com.nate.android.portalmini", "com.nhn.android.moneybook", "com.nhn.android.nmap",
                                "com.nhn.land.android", "com.nhnent.payapp", "com.nike.plusgps",
                                "com.northcube.sleepcycle", "com.nsmobilehub", "com.olleh.android.oc2",
                                "com.omnitel.android.lottewebview", "com.org.yakult", "com.pione.questiondiary",
                                "com.popularapp.periodcalendar", "com.pub.fm", "com.rainist.banksalad2",
                                "com.realbyteapps.moneymanagerfree", "com.sampleapp", "com.samsung.android.oneconnect",
                                "com.samsung.android.spay", "com.samsung.ssmobile", "com.samsungfire.insurance",
                                "com.samsungpop.android.mpop", "com.sec.android.app.shealth", "com.shcard.smartpay",
                                "com.shilladutyfree", "com.shinhan.sbanking", "com.shinhancard.smartshinhan",
                                "com.skmc.okcashbag.home_google", "com.skplanet.weatherpong.mobile", "com.skt.aladdin",
                                "com.skt.logii.widget", "com.skt.skaf.OA00026910", "com.skt.smartbill",
                                "com.sktelecom.tauth", "com.sktelecom.tsmartpay", "com.smartro.gongcha", "com.smg.spbs",
                                "com.socialapps.homeplus", "com.soilbonus.goodoilfamily", "com.sportractive",
                                "com.ssg.serviceapp.android.egiftcertificate", "com.starbucks.co", "com.strava",
                                "com.tenfingers.seouldatepop", "com.tinder", "com.tmon", "com.tms", "com.ton.tarotofoz",
                                "com.tourbaksaapp.app", "com.tourlive", "com.towneers.www", "com.uniqlo.kr.catalogue",
                                "com.unsin", "com.uplus.baseballhdtv", "com.weeds.tillionpanel.full", "com.wemakeprice",
                                "com.whox2.lguplus", "com.wooribank.pib.smart", "com.wooribank.pot.smart",
                                "com.wooribank.smart.mwib", "com.wooribank.smart.wwms", "com.wooricard.smartapp",
                                "com.wooriwm.mugsmart", "com.wooriwm.txsmart", "com.wr.alrim", "com.xiaomi.hm.health",
                                "com.xiaomi.smarthome", "com.yuanta.helloyuanta", "com.yuanta.tradarm",
                                "de.flixbus.app", "handasoft.mobile.divination", "hyundai.hi.smart.android.activity",
                                "kr.ac.kaist.bab", "kr.backpackr.me.idus", "kr.co.bitsonic", "kr.co.burgerkinghybrid",
                                "kr.co.cjone.app.lockscreen", "kr.co.company.hwahae", "kr.co.emart.emartmall",
                                "kr.co.gscaltex.gsnpoint", "kr.co.hanamembers.hmscustomer",
                                "kr.co.ivlog.mobile.app.cjonecard", "kr.co.keypair.keywalletTouch",
                                "kr.co.kiwiplus.kiwi.kiwimom", "kr.co.lgfashion.lgfashionshop.v28",
                                "kr.co.mercurykorea", "kr.co.openit.openrider", "kr.co.psynet",
                                "kr.co.samsungcard.mpocket", "kr.co.srail.app", "kr.co.ssg", "kr.or.kftc.mobileapt2you",
                                "kt.co.ktmmobile", "kvp.jjy.MispAndroid320",
                                "losebellyfat.flatstomach.absworkout.fatburning",
                                "lt.spectrofinance.spectrocoin.android.wallet", "net.daum.android.map",
                                "net.giosis.shopping.sg", "net.ib.android.smcard", "net.skyscanner.android.main",
                                "nh.smart.signone", "sam.myanycar.samsungFire", "sixpack.sixpackabs.absworkout",
                                "sk.com.shopping", "stonykids.baedaltong.season2", "uplus.membership",
                                "viva.republica.toss", "com.SunHan", "com.WeTogether", "com.android.calendar",
                                "com.android.hotwordenrollment.okgoogle", "com.gdragon.pod.leeson", "com.kakao.home",
                                "com.kt.olleh.storefront", "com.lge.sizechangable.weather", "com.luga",
                                "com.samsung.android.calendar", "com.samsung.android.samsungpay.gear",
                                "com.samsung.android.weather", "com.sec.android.app.billing",
                                "com.sec.android.app.clockpackage", "com.sec.android.app.samsungapps",
                                "com.shinhan.global.vn.sclub", "com.show.greenbill", "com.skt.skaf.A000Z00040",
                                "com.skt.skaf.OA00018282", "com.skt.skaf.OA00199800", "com.smhk.boomerang",
                                "kr.ac.kaist.kyotong", "kr.co.kbo.redss.app", "kr.pe.javarss.barcodewallet2",
                                "net.daum.android.solcalendar", "net.giosis.shopping.hub", "nh.smart"],
                  "AUTO_AND_VEHICLES": ["ch.sbb.mobile.android.b2c", "com.astroframe.seoulbus", "com.djbus",
                                        "com.dwlife.drinkingcolor", "com.ebcard.bustago", "com.encar.encarMobileApp",
                                        "com.encardirect.app", "com.google.android.projection.gearhead", "com.greencar",
                                        "com.kakao.cp.driver", "com.kakao.taxi", "com.kakao.wheel.driver",
                                        "com.kscc.scxb.mbl", "com.kscc.xzz.mbl", "com.kt.mtmoney",
                                        "com.locnall.KimGiSa", "com.nbdproject.macarong", "com.skn.zamong2",
                                        "com.skt.tmap.ku", "com.ubercab", "de.hafas.android.db", "kr.co.cardoc",
                                        "kr.co.kbc.cha.android", "kr.co.vcnc.tada", "kt.navi", "net.orizinal.subway",
                                        "socar.Socar", "teamDoppelGanger.SmarterSubway",
                                        "com.httpdongbu_directcar.dbdirect", "com.skt.skaf.l001mtm091"],
                  "EDUCATION": ["com.Classting", "com.belugaedu.amgigorae", "com.brainbow.peak.app",
                                "com.carrotenglish.rptest", "com.hellotalk", "com.imcompany.school2",
                                "com.kcic.oneminjapanese", "com.mathpresso.qandateacher", "com.moiseum.dailyart2",
                                "com.naver.naveraudio", "com.nhn.android.naverdic", "com.qualson.realclass",
                                "com.stn.mobile_player", "kr.ac.kaist.coursemos.android", "kr.co.sigongedu.memo",
                                "kr.co.yanadoo.mobile", "net.carrotenglish.ctm", "com.diotek.sec.lookup.dictionary",
                                "com.omega.dic.es", "com.sec.android.app.dictionary"],
                  "GAME": ["com.Monthly23.SwipeBrickBreaker", "com.bandainamcoent.onepiecetresurecruisekr",
                           "com.kakaogames.friendsKing", "com.kakaogames.friendsTower", "com.kakaogames.gnss",
                           "com.nexon.da3", "com.nexon.nsc.maplem", "com.outfit7.mytalkingtomfree",
                           "com.pearlabyss.blackdesertm", "com.playrix.gardenscapes", "com.pubg.krmobile",
                           "com.puzzlegames.collection.puzzle", "com.sundaytoz.kakao.anipang3.service",
                           "com.sundaytoz.kakao.wbb", "com.supercell.clashroyale", "com.tinyco.potter",
                           "com.wellgames.ss", "kr.aos.com.aprogen.hng.fortressm", "com.samsung.android.game.gamehome",
                           "com.samsung.android.game.gametools"],
                  "PRODUCTIVITY_AND_BUSINESS": ["com.Slack", "com.adobe.reader", "com.albamon.app",
                                                "com.alphainventor.filemanager", "com.androidrocker.callblocker",
                                                "com.credu.ml", "com.dropbox.android",
                                                "com.estmob.android.sendanywhere", "com.evernote",
                                                "com.google.android.apps.ads.publisher", "com.google.android.apps.docs",
                                                "com.google.android.apps.docs.editors.docs",
                                                "com.google.android.apps.docs.editors.sheets",
                                                "com.google.android.apps.docs.editors.slides",
                                                "com.google.android.apps.m4b", "com.google.android.calendar",
                                                "com.google.android.keep", "com.infraware.office.link",
                                                "com.intsig.camscanner", "com.jobkorea.app", "com.jobplanet.kr.android",
                                                "com.jushine.cstandard", "com.microsoft.office.excel",
                                                "com.microsoft.office.onenote", "com.microsoft.office.powerpoint",
                                                "com.microsoft.office.word", "com.microsoft.rdc.android",
                                                "com.microsoft.skydrive", "com.naver.nozzle",
                                                "com.nhn.android.navermemo", "com.nhn.android.ndrive",
                                                "com.samsung.android.app.notes", "com.samsung.android.email.provider",
                                                "com.samsung.android.galaxycontinuity", "com.samsung.knox.securefolder",
                                                "com.sec.android.inputmethod", "com.skt.prod.cloud",
                                                "com.sktelecom.tguard", "com.socialnmobile.dictapps.notepad.color.note",
                                                "com.surpax.ledflashlight.panel", "com.sweettracker.smartparcel",
                                                "com.teamup.teamup", "com.teamviewer.teamviewer.market.mobile",
                                                "com.tf.thinkdroid.viewer", "com.truemobile.ipdisk",
                                                "kr.co.alba.webappalba.m", "kr.co.hiworks.messenger",
                                                "kr.co.hiworks.mobile", "kr.co.rememberapp", "kr.co.rinasoft.howuse",
                                                "kr.co.saramin.SalaryCalc", "lg.uplusbox", "net.slideshare.mobile",
                                                "org.isoron.uhabits", "com.hancom.office.editor.sec",
                                                "com.imoxion.sensmobile", "com.linkedin.android.jobs.jobseeker",
                                                "com.noknok.android.mfac.service", "com.nsds.myphoneandme"],
                  "TOOLS": ["com.ahnlab.v3mobileplus", "com.ahnlab.v3mobilesecurity.soda", "com.asus.remotelink.full",
                            "com.atsolution.android.uotp2", "com.cleanmaster.mguard", "com.digibites.accubattery",
                            "com.estsoft.alyac", "com.fiberthemax.OpQ2keyboard", "com.finedigital.finewifiremocon",
                            "com.google.android.apps.authenticator2", "com.google.android.apps.translate",
                            "com.google.android.calculator", "com.google.android.gms",
                            "com.google.android.googlequicksearchbox", "com.lge.lgpay", "com.lonelycatgames.Xplore",
                            "com.meonria.scientificcalc", "com.mi.android.globalFileexplorer", "com.miui.calculator",
                            "com.mobidia.android.mdm", "com.naver.labs.translator", "com.necta.wifimousefree",
                            "com.samsung.android.app.watchmanager", "com.samsung.android.geargplugin",
                            "com.samsung.android.gearoplugin", "com.samsung.android.lool",
                            "com.samsung.android.mobileservice", "com.samsung.android.sidegesturepad",
                            "com.samsung.android.voc", "com.sec.android.app.myfiles",
                            "com.sec.android.app.popupcalculator", "com.sec.android.app.voicenote",
                            "com.sec.android.easyMover", "com.skt.prod.phonebook", "org.cohortor.gstrings",
                            "org.swadhin.app", "android.example.com.tflitecamerademo",
                            "catchpower.gogo.blackholeaddons", "com.android.apps.tag", "com.android.bluetooth",
                            "com.android.calculator2", "com.android.captiveportallogin",
                            "com.android.companiondevicemanager", "com.android.contacts", "com.android.documentsui",
                            "com.android.htmlviewer", "com.android.nfc", "com.android.phone",
                            "com.android.providers.downloads", "com.android.providers.media",
                            "com.android.server.telecom", "com.android.settings", "com.android.soundrecorder",
                            "com.android.updater", "com.android.vending", "com.clip.getapp", "com.edgeblock.edgeblock",
                            "com.facebook.appmanager", "com.fingerprints.fido.asm",
                            "com.google.android.packageinstaller", "com.google.android.setupwizard", "com.lge.clock",
                            "com.lge.phonemanagement", "com.lge.qhelp", "com.lge.qmemoplus", "com.lge.qremote",
                            "com.lge.shutdownmonitor", "com.lge.smartcover", "com.lge.smartsharepush",
                            "com.lge.updatecenter", "com.lge.voicerecorder", "com.lge.wifisettings", "com.samsung.SMT",
                            "com.samsung.android.MtpApplication", "com.samsung.android.app.advsounddetector",
                            "com.samsung.android.app.galaxyfinder", "com.samsung.android.app.memo",
                            "com.samsung.android.app.reminder", "com.samsung.android.app.scrollcapture",
                            "com.samsung.android.app.smartcapture", "com.samsung.android.app.soundpicker",
                            "com.samsung.android.app.taskedge", "com.samsung.android.authfw",
                            "com.samsung.android.bixby.agent", "com.samsung.android.bixby.wakeup",
                            "com.samsung.android.contacts", "com.samsung.android.da.daagent",
                            "com.samsung.android.edgelightingeffectunit", "com.samsung.android.goodlock",
                            "com.samsung.android.mateagent", "com.samsung.android.pluginrecents",
                            "com.samsung.android.samsungpass", "com.samsung.android.samsungpassautofill",
                            "com.samsung.android.scloud", "com.samsung.android.sconnect",
                            "com.samsung.android.themestore", "com.samsung.app.highlightplayer",
                            "com.sec.android.app.taskmanager", "com.sec.android.app.vepreload",
                            "com.sec.android.daemonapp", "com.sec.android.easyMover.Agent",
                            "com.sec.android.mimage.photoretouching", "com.sec.android.wallpapercropper2",
                            "com.skp.clink.invoke", "com.skt.prod.phone", "com.skt.prod.tphonelite",
                            "com.skt.t_smart_charge", "com.sonyericsson.pws", "com.sonyericsson.soundenhancement",
                            "com.sonymobile.android.contacts", "com.tarkarn.denialcall", "com.tv.remote.controletv.tv",
                            "com.vlingo.midas", "com.wssyncmldm", "devian.tubemate.v3", "org.tensorflow.demo"],
                  "PHOTOGRAPHY": ["com.alensw.PicFolder", "com.artifyapp.timestamp", "com.campmobile.snow",
                                  "com.commsource.beautyplus", "com.cyworld.camera", "com.fotoable.enstyle",
                                  "com.google.android.apps.photos", "com.kakao.cheez", "com.linecorp.b612.android",
                                  "com.linecorp.foodcam.android", "com.lyrebirdstudio.collage", "com.oss.mcam",
                                  "com.photofunia.android", "com.sec.android.gallery3d", "com.sonyericsson.album",
                                  "com.sonymobile.moviecreator.rmm", "com.ster.photo.surgery", "ph.app.instasave",
                                  "photoeditor.layout.collagemaker", "com.android.camera", "com.android.gallery3d",
                                  "com.aviary.android.feather", "com.lge.camera", "com.miui.gallery",
                                  "com.samsung.android.visionintelligence", "com.sec.android.app.camera",
                                  "com.sec.android.app.camera.avatarauth", "com.sonyericsson.android.camera"],
                  "SOCIAL_AND_COMMUNICATION": ["com.andr.evine.who", "com.android.chrome",
                                               "com.briniclemobile.wibeetalk", "com.btb.minihompy",
                                               "com.buzzvil.adhours", "com.cashslide", "com.dcinside.app",
                                               "com.discord", "com.enabledaonsoft.thecamp", "com.facebook.katana",
                                               "com.facebook.orca", "com.google.android.apps.messaging",
                                               "com.google.android.gm", "com.google.android.talk",
                                               "com.instagram.android", "com.kakao.story", "com.kakao.talk",
                                               "com.kickstarter.kickstarter", "com.linkedin.android",
                                               "com.nexon.nxplay", "com.nhn.android.band", "com.nhn.android.blog",
                                               "com.nhn.android.mail", "com.nhn.android.navercafe", "com.opera.browser",
                                               "com.ppomppu.android", "com.samsung.accessory",
                                               "com.sec.android.app.sbrowser", "com.sec.spp.push",
                                               "com.skt.prod.dialer", "com.skt.tdatacoupon", "com.skype.raider",
                                               "com.snapchat.android", "com.sonelli.juicessh", "com.teamblind.blind",
                                               "com.tencent.mm", "com.tumblr", "com.vaultmicro.kidsnote",
                                               "com.vidyo.VidyoClient", "com.whatsapp", "gogolook.callgogolook2",
                                               "jp.naver.line.android", "kr.ac.kaist.portal",
                                               "kr.co.vcnc.android.couple", "kreditdata.kreditjob",
                                               "net.daum.android.cafe", "net.daum.android.daum",
                                               "net.daum.android.tistoryapp", "org.telegram.messenger",
                                               "com.android.browser", "com.android.incallui", "com.android.mms",
                                               "com.samsung.android.incallui", "com.samsung.android.messaging",
                                               "com.samsung.android.service.livedrawing",
                                               "com.sonymobile.android.dialer", "kr.co.vcnc.betweendate.date",
                                               "net.hibrain.apps.android.hibrainnet"],
                  "ENTERTAINMENT": ["com.appgate.gorealra", "com.camobile.akb48mail", "com.cgv.android.movieapp",
                                    "com.delphicoder.flud", "com.frograms.watcha",
                                    "com.google.android.apps.youtube.kids", "com.google.android.play.games",
                                    "com.google.android.videos", "com.google.android.youtube", "com.imbc.mini",
                                    "com.interpark.app.ticket", "com.kt.otv", "com.megabox.mop", "com.morp.arirangVV",
                                    "com.mxtech.videoplayer.ad", "com.mxtech.videoplayer.pro", "com.naver.vapp",
                                    "com.netflix.mediaclient", "com.nexon.handsplus", "com.nhn.android.naverplayer",
                                    "com.qiyi.video.pad", "com.quvideo.xiaoying",
                                    "com.samsung.everland.android.mobileApp", "com.scatterlab.sol", "com.skb.btvmobile",
                                    "com.skb.smartrc", "com.skmnc.gifticon", "com.sonyericsson.xhs",
                                    "gg.op.lol.android", "kr.co.captv.pooqV2", "kr.co.kbs.kong",
                                    "kr.co.lottecinema.lcm", "kr.co.ticketlink.cne",
                                    "org.leetzone.android.yatsewidgetfree", "tv.jamlive", "tv.twitch.android.app",
                                    "video.player.videoplayer", "zettamedia.bflix", "com.samsung.android.video",
                                    "com.sec.android.app.dmb", "com.sec.android.app.videoplayer", "sixclk.newpiki"],
                  "MUSIC_AND_AUDIO": ["com.appmind.radios.jp", "com.behringer.android.control.app.m32q",
                                      "com.cocoradio.country.jp", "com.fundevs.app.mediaconverter",
                                      "com.gomtv.gomaudio", "com.google.android.apps.youtube.music",
                                      "com.google.android.music", "com.iloen.melon", "com.jaybirdsport.audio",
                                      "com.kakao.music", "com.mnet.app", "com.naver.vibe", "com.nhn.android.music",
                                      "com.sec.android.app.music", "com.shazam.android", "com.soribada.android",
                                      "com.soundbrenner.pulse", "fm.qingting.qtradio", "kr.ebs.bandi",
                                      "radiotime.player", "skplanet.musicmate", "tunein.player", "com.lge.music",
                                      "com.samsung.radio", "com.samsung.voiceserviceplatform",
                                      "com.sec.android.app.soundalive"],
                  "BOOKS_AND_REFERENCE": ["com.bambuna.podcastaddict", "com.clem.nhkradio",
                                          "com.cnn.mobile.android.phone", "com.godpeople.GPBIBLE",
                                          "com.initialcoms.ridi", "com.kakao.page", "com.kyobo.ebook.samsung",
                                          "com.medium.reader", "com.nhn.android.nbooks", "com.nhn.android.search",
                                          "com.nhn.android.webtoon", "com.ptculi.tekview", "com.skt.skaf.OA00050017",
                                          "com.sony.nfx.app.sfrc", "com.twitter.android", "flipboard.boxer.app",
                                          "fm.castbox.audiobook.radio.podcast", "kr.ac.libit.kaist",
                                          "kr.co.aladin.ebook", "kr.co.ypbooks.app", "mok.android",
                                          "net.daum.android.webtoon", "org.wikipedia", "com.sktechx.xviewer"]}

APP_LAUNCHER = [
    'com.sec.android.app.launcher', 'com.lge.launcher3', 'com.google.android.apps.nexuslauncher', 'com.android.systemui'
]

APP_EXPERIMENT = ['com.fitbit.FitbitMobile', 'kaist.iclab.abc']

PERIODS_FORMAT = [
    ('YR', 60 * 60 * 24 * 365),
    ('MONTH', 60 * 60 * 24 * 30),
    ('DAY', 60 * 60 * 24),
    ('HR', 60 * 60),
    ('MIN', 60),
    ('SEC', 1)
]

WEEKDAYS = [
    'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'
]


def __format_window__(size_sec: int):
    for form, seconds in PERIODS_FORMAT:
        if size_sec >= seconds:
            val, _ = divmod(size_sec, seconds)
            return '{}-{}'.format(val, form)

    return '{}-{}'.format(size_sec, 'SEC')


def __extract_window_data__(data: pd.DataFrame, from_millis: int, to_millis: int, begin_col: str,
                            end_col: Optional[str] = None) -> pd.DataFrame:
    # For begin - end column, from_millis < end_col & to_millis > begin_col
    # For single point, from_millis <= begin_col < to_millis

    if end_col is None:
        ret = data.loc[lambda x: (from_millis <= x[begin_col]) & (x[begin_col] < to_millis), :]
    else:
        ret = data.loc[lambda x: (from_millis < x[end_col]) | (to_millis > x[begin_col]), :]
    return ret.copy()


def __extract_immediate_past_data__(data: pd.DataFrame,
                                    point_millis: int, col: str) -> Optional[pd.Series]:
    try:
        return data.loc[lambda x: x[col] < point_millis, :].iloc[-1]
    except IndexError:
        return None


def __duration__(from_array: np.ndarray, to_array: np.ndarray,
                 from_t: Union[int, float], to_t: Union[int, float], win_size: int) -> np.ndarray:
    non_na_to = np.nan_to_num(to_array, nan=to_t)
    non_na_from = np.nan_to_num(from_array, nan=from_t)

    return np.where(
        non_na_to > to_t, to_t, to_array
    ) - np.where(
        non_na_from < from_t, from_t, from_array
    ) / (win_size * 1000)


def __dwt_duration__(min_time: Union[int, float],
                     max_time: Union[int, float],
                     timestamps: np.ndarray,
                     duration: np.ndarray,
                     step_size: int = 256,
                     n_coeffs: int = 32) -> np.ndarray:
    assert timestamps.shape[0] == duration.shape[0]

    if timestamps.shape[0] == 0:
        return np.zeros(n_coeffs)

    max_duration = (max_time - min_time) / step_size
    sample = np.linspace(min_time, max_time - 1, step_size)
    time_series = np.sort(np.hstack([timestamps, timestamps + duration]))
    bin_indices = np.digitize(sample, time_series)
    bin_values = np.take(time_series, bin_indices - 1)
    step_duration = np.roll(
        np.where(sample - bin_values >= max_duration, max_duration, bin_values - sample + max_duration), -1
    )
    filtered_duration = np.where(
        np.mod(bin_indices, 2) == 1, step_duration, 0
    )

    return np.hstack(wavedec(filtered_duration, 'haar'))[:n_coeffs]


def __dwt_numeric__(min_time: Union[int, float],
                    max_time: Union[int, float],
                    timestamps: np.ndarray,
                    values: np.ndarray,
                    step_size: int = 256,
                    n_coeffs: int = 32) -> np.ndarray:
    assert timestamps.shape[0] == values.shape[0]

    if timestamps.shape[0] == 0:
        return np.zeros(n_coeffs)

    sample = np.linspace(min_time, max_time - 1, step_size)
    interpolates = np.interp(sample, timestamps, values)

    return np.hstack(wavedec(interpolates, 'haar'))[:n_coeffs]


def __mean__(d: pd.Series, na: Union[int, float] = 0) -> Union[int, float]:
    mean = d.mean()
    if np.isnan(mean):
        mean = na
    return mean


def __std__(d: pd.Series, na: Union[int, float] = -1) -> Union[int, float]:
    std = d.std()
    if np.isnan(std):
        std = na
    return std


class DataProcessor(ABC):
    def __init__(self, windows: Optional[List[int]] = None):
        self._windows = windows

    @classmethod
    def __to_feature__(cls, ts: int, win_size: int,
                       name: str,
                       value: Optional[Union[str, float]],
                       categories: Optional[Union[List[str], List[bool]]] = None) -> Feature:

        win_name = 'CUR' if win_size <= 0 else __format_window__(win_size)
        return Feature(ts, '{}_{}'.format(win_name, name), value, categories)

    @abstractmethod
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        pass

    @classmethod
    @abstractmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        pass

    async def generate_features(self,
                                data: pd.DataFrame,
                                data_points: List[int]) -> Tuple[pd.DataFrame, Dict[str, Union[Set[str], Set[bool]]]]:
        _prepare = perf_counter()
        prepared_data = await self.__prepare_data__(data)

        print('{} Prepare complete: {:.5f} s'.format(self.__class__.__name__, perf_counter() - _prepare))

        tasks = []

        for data_point in data_points:
            tasks.append(
                self.__generate_current_feature__(data_point, prepared_data)
            )

            if self._windows:
                for window in self._windows:
                    tasks.append(
                        self.__generate_window_feature__(data_point, window, prepared_data)
                    )
        _feature = perf_counter()
        result = await asyncio.gather(*tasks)
        print('{} Feature generation complete: {:.5f} s'.format(self.__class__.__name__, perf_counter() - _feature))

        _flatten = perf_counter()

        flatten_features = [(item.timestamp, item.name, item.value) for sub_list in result for item in sub_list]

        ret_data = pd.DataFrame(flatten_features,
                                columns=['timestamp', 'feature', 'value']
                                ).pivot(index='timestamp', columns='feature', values='value')
        category_info = dict()

        for sub_list in result:
            for item in sub_list:
                if item.categories:
                    if category_info.get(item.name) is None:
                        category_info[item.name] = set(item.categories)
                    else:
                        category_info[item.name] = category_info[item.name].union(item.categories)

        return ret_data, category_info


async def process_data(participant: str,
                       raw_data: List[Tuple[pd.DataFrame, DataProcessor]],
                       class_label: pd.DataFrame,
                       class_label_on: str) -> ProcessedData:
    workers = []

    for raw_datum, processor in raw_data:
        workers.append(
            processor.generate_features(raw_datum, class_label.loc[:, class_label_on])
        )

    total_results = await asyncio.gather(*workers)
    processed_data = None
    categories = dict()

    for result, category in total_results:
        if processed_data is None:
            processed_data = result
        else:
            processed_data = pd.merge(processed_data, result, on='timestamp')

        categories = {
            **categories,
            **category
        }

    merged_data = pd.merge(
        processed_data, class_label, left_on='timestamp', right_on=class_label_on, how='inner'
    )

    class_label_only = [label for label in class_label.columns if label != class_label_on]

    train = merged_data.loc[:, lambda x: ~x.columns.isin(class_label.columns)]
    label = merged_data.loc[:, lambda x: x.columns.isin(class_label_only)]
    categories = [
        (np.flatnonzero(train.columns.isin([name]))[0], values)
        for name, values in categories.items()
    ]

    return ProcessedData(
        id=participant,
        train=train,
        label=label,
        categories=categories
    )


class AppUsageProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        app_data = data.loc[lambda x: x['type'].isin(['MOVE_TO_FOREGROUND', 'MOVE_TO_BACKGROUND']), :]

        concat_data = pd.concat([
            app_data,
            app_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        return []

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )

        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[
            lambda x: (x['duration'] > 0) & (x['type'] == 'MOVE_TO_FOREGROUND'),
            ['timestamp', 'package_name', 'duration']
        ]

        app_switching_frequency = len(window_data.index) / win_size
        app_switching_frequency_wo_launcher = len(window_data.loc[
                                                  lambda x: ~x['package_name'].isin(APP_LAUNCHER), :].index
                                                  ) / win_size
        features += [
            cls.__to_feature__(
                data_point, win_size, 'APP_SWITCHING_FREQUENCY', app_switching_frequency
            ),
            cls.__to_feature__(
                data_point, win_size, 'APP_SWITCHING_FREQUENCY_WO_LAUNCHER', app_switching_frequency_wo_launcher
            )
        ]

        for category, apps in APP_CATEGORIES.items():
            category_data = window_data.loc[lambda x: x['package_name'].isin(apps), :]

            duration_mean = __mean__(category_data.loc[:, 'duration'])
            duration_std = __std__(category_data.loc[:, 'duration'])
            mean_frequency = len(category_data.index) / win_size

            features += [
                cls.__to_feature__(data_point, win_size, 'APP_{}_DURATION_MEAN'.format(category), duration_mean),
                cls.__to_feature__(data_point, win_size, 'APP_{}_DURATION_STD'.format(category), duration_std),
                cls.__to_feature__(data_point, win_size, 'APP_{}_FREQUENCY'.format(category), mean_frequency)
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    category_data.loc[:, 'timestamp'].values,
                    category_data.loc[:, 'duration'].values,
                )

                features += [
                    cls.__to_feature__(data_point,
                                       win_size,
                                       'APP_{}_DURATION_DWT_{:02d}_COEFF'.format(category, idx),
                                       coeff)
                    for idx, coeff in enumerate(dwt)
                ]

        return features


class BatteryProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        diff_data = data.loc[lambda x: x['plugged'] != x.shift(1)['plugged'], :]
        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(
            data, millis, 'timestamp'
        )

        if past_data is not None:
            return [
                cls.__to_feature__(data_point, 0, 'BATTERY_LEVEL', past_data['level']),
                cls.__to_feature__(data_point, 0, 'BATTERY_TEMPERATURE', past_data['temperature']),
                cls.__to_feature__(data_point, 0, 'BATTERY_PLUGGED', past_data['plugged'] != 'UNDEFINED', [True, False])
            ]

        return []

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(data, from_millis, to_millis, 'timestamp', '_timestamp')

        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )
        plugged_duration_data = window_data.loc[
            lambda x: (x['duration'] > 0) & (x['plugged'] != 'UNDEFINED'), ['timestamp', 'duration']
        ]
        unplugged_duration_data = window_data.loc[
            lambda x: (x['duration'] > 0) & (x['plugged'] == 'UNDEFINED'), ['timestamp', 'duration']
        ]
        level_data = window_data.loc[:, ['timestamp', 'level']]
        temperature_data = window_data.loc[:, ['timestamp', 'temperature']]

        plugged_mean_frequency = len(plugged_duration_data.index) / win_size
        unplugged_mean_frequency = len(unplugged_duration_data.index) / win_size
        plugged_duration_mean = __mean__(plugged_duration_data.loc[:, 'duration'])
        plugged_duration_std = __std__(plugged_duration_data.loc[:, 'duration'])
        unplugged_duration_mean = __mean__(unplugged_duration_data.loc[:, 'duration'])
        unplugged_duration_std = __std__(unplugged_duration_data.loc[:, 'duration'])
        level_mean = __mean__(level_data.loc[:, 'level'])
        level_std = __std__(level_data.loc[:, 'level'])
        temperature_mean = __mean__(temperature_data.loc[:, 'temperature'])
        temperature_std = __std__(temperature_data.loc[:, 'temperature'])

        features += [
            cls.__to_feature__(data_point, win_size, 'BATTERY_PLUGGED_FREQUENCY', plugged_mean_frequency),
            cls.__to_feature__(data_point, win_size, 'BATTERY_UNPLUGGED_FREQUENCY', unplugged_mean_frequency),
            cls.__to_feature__(data_point, win_size, 'BATTERY_PLUGGED_DURATION_MEAN', plugged_duration_mean),
            cls.__to_feature__(data_point, win_size, 'BATTERY_PLUGGED_DURATION_STD', plugged_duration_std),
            cls.__to_feature__(data_point, win_size, 'BATTERY_UNPLUGGED_DURATION_MEAN', unplugged_duration_mean),
            cls.__to_feature__(data_point, win_size, 'BATTERY_UNPLUGGED_DURATION_STD', unplugged_duration_std),
            cls.__to_feature__(data_point, win_size, 'BATTERY_LEVEL_MEAN', level_mean),
            cls.__to_feature__(data_point, win_size, 'BATTERY_LEVEL_STD', level_std),
            cls.__to_feature__(data_point, win_size, 'BATTERY_TEMPERATURE_MEAN', temperature_mean),
            cls.__to_feature__(data_point, win_size, 'BATTERY_TEMPERATURE_STD', temperature_std)
        ]

        if win_size >= 60:
            dwt_plugged_duration = __dwt_duration__(
                from_millis, to_millis,
                plugged_duration_data.loc[:, 'timestamp'].values,
                plugged_duration_data.loc[:, 'duration'].values
            )
            dwt_unplugged_duration = __dwt_duration__(
                from_millis, to_millis,
                unplugged_duration_data.loc[:, 'timestamp'].values,
                unplugged_duration_data.loc[:, 'duration'].values
            )
            dwt_level = __dwt_numeric__(
                from_millis, to_millis,
                level_data.loc[:, 'timestamp'].values,
                level_data.loc[:, 'level'].values,
            )
            dwt_temperature = __dwt_numeric__(
                from_millis, to_millis,
                temperature_data.loc[:, 'timestamp'].values,
                temperature_data.loc[:, 'temperature'].values,
            )

            features += [
                cls.__to_feature__(data_point, win_size, 'BATTERY_PLUGGED_DURATION_DWT_{:02d}_COEFF'.format(idx), coeff)
                for idx, coeff in enumerate(dwt_plugged_duration)
            ]
            features += [
                cls.__to_feature__(data_point, win_size, 'BATTERY_UNPLUGGED_DURATION_DWT_{:02d}_COEFF'.format(idx),
                                   coeff)
                for idx, coeff in enumerate(dwt_unplugged_duration)
            ]
            features += [
                cls.__to_feature__(data_point, win_size, 'BATTERY_LEVEL_DWT_{:02d}_COEFF'.format(idx), coeff)
                for idx, coeff in enumerate(dwt_level)
            ]
            features += [
                cls.__to_feature__(data_point, win_size, 'BATTERY_TEMPERATURE_DWT_{:02d}_COEFF'.format(idx), coeff)
                for idx, coeff in enumerate(dwt_temperature)
            ]

        return features


class CallLogProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_with_start_time = data.assign(
            _timestamp=lambda x: x['timestamp'] - x['duration'] * 1000
        )
        return data_with_start_time

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        millis = data_point * 1000
        is_calling = len(data.loc[lambda x: (x['timestamp'] < millis) & (x['timestamp'] > millis), :].index) != 0

        return [cls.__to_feature__(data_point, 0, 'CALL_IN_PROGRESS', is_calling, [True, False])]

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, '_timestamp', 'timestamp'
        )
        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, '_timestamp'].values, window_data.loc[:, 'timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[
            lambda x: (x['duration'] > 0) | (x['type'] == 'MISSED'),
            ['_timestamp', 'duration', 'type', 'contact']
        ]

        for call_type, is_contact in list(product(['INCOMING', 'OUTGOING', 'MISSED'], [True, False])):
            sub_data = window_data.loc[
                       lambda x: (x['type'] == call_type) &
                                 (x['contact'] != 'UNDEFINED' if is_contact else x['contact'] == 'UNDEFINED'),
                       :]
            contact = 'CONTACT' if is_contact else 'NON_CONTACT'
            mean_frequency = len(sub_data.index) / win_size
            features += [
                cls.__to_feature__(
                    data_point, win_size, 'CALL_{}_{}_FREQUENCY'.format(call_type, contact), mean_frequency
                )
            ]

            if call_type != 'MISSED':
                duration_mean = __mean__(sub_data.loc[:, 'duration'])
                duration_std = __std__(sub_data.loc[:, 'duration'])

                features += [
                    cls.__to_feature__(
                        data_point, win_size, 'CALL_{}_{}_DURATION_MEAN'.format(call_type, contact), duration_mean
                    ),
                    cls.__to_feature__(
                        data_point, win_size, 'CALL_{}_{}_DURATION_STD'.format(call_type, contact), duration_std
                    )
                ]

                if win_size >= 60:
                    dwt_duration = __dwt_duration__(
                        from_millis, to_millis,
                        sub_data.loc[:, '_timestamp'].values,
                        sub_data.loc[:, 'duration'].values
                    )
                    features += [
                        cls.__to_feature__(
                            data_point, win_size,
                            'CALL_{}_{}_DURATION_DWT_{:02d}_COEFF'.format(call_type, contact, idx), coeff
                        ) for idx, coeff in enumerate(dwt_duration)
                    ]

        return features


class ConnectivityProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        shift_data = data.shift(1)
        diff_data = data.loc[lambda x: x['type'] != shift_data['type']]

        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(data, millis, 'timestamp')

        if past_data is not None:
            conn_types = ['WIFI', 'MOBILE', 'UNDEFINED']
            conn_type = 'UNDEFINED'

            for t in conn_types:
                if past_data['type'].startswith(t):
                    conn_type = past_data['type']
                    break

            features += [cls.__to_feature__(data_point, 0, 'CONN_TYPE', conn_type, ['WIFI', 'MOBILE', 'UNDEFINED'])]

        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )
        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[lambda x: x['duration'] > 0, ['timestamp', 'duration', 'type']]

        for conn_type in ['WIFI', 'MOBILE', 'UNDEFINED']:
            sub_data = window_data.loc[lambda x: x['type'].str.startswith(conn_type), :]
            mean_frequency = len(sub_data.index) / win_size
            duration_mean = __mean__(sub_data.loc[:, 'duration'])
            duration_std = __std__(sub_data.loc[:, 'duration'])

            features += [
                cls.__to_feature__(data_point, win_size, 'CONN_{}_DURATION_MEAN'.format(conn_type), duration_mean),
                cls.__to_feature__(data_point, win_size, 'CONN_{}_DURATION_STD'.format(conn_type), duration_std),
                cls.__to_feature__(data_point, win_size, 'CONN_{}_FREQUENCY'.format(conn_type), mean_frequency)
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    sub_data.loc[:, 'timestamp'].values,
                    sub_data.loc[:, 'duration'].values
                )
                features += [
                    cls.__to_feature__(
                        data_point, win_size, 'CONN_{}_DURATION_DWT_{:02d}_COEFF'.format(conn_type, idx), coeff
                    ) for idx, coeff in enumerate(dwt)
                ]

        return features


class DataTrafficProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        return []

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp'
        )

        rx_data = window_data.loc[lambda x: x['rx_kb'] > 0, ['timestamp', 'rx_kb']]
        tx_data = window_data.loc[lambda x: x['tx_kb'] > 0, ['timestamp', 'tx_kb']]

        rx_mean = __mean__(rx_data.loc[:, 'rx_kb'])
        rx_std = __std__(rx_data.loc[:, 'rx_kb'])
        tx_mean = __mean__(tx_data.loc[:, 'tx_kb'])
        tx_std = __std__(tx_data.loc[:, 'tx_kb'])

        features += [
            cls.__to_feature__(data_point, win_size, 'DATA_RX_MEAN', rx_mean),
            cls.__to_feature__(data_point, win_size, 'DATA_RX_STD', rx_std),
            cls.__to_feature__(data_point, win_size, 'DATA_TX_MEAN', tx_mean),
            cls.__to_feature__(data_point, win_size, 'DATA_TX_STD', tx_std)
        ]

        if win_size >= 60:
            dwt_rx = __dwt_numeric__(
                from_millis, to_millis,
                rx_data.loc[:, 'timestamp'].values,
                rx_data.loc[:, 'rx_kb'].values
            )
            dwt_tx = __dwt_numeric__(
                from_millis, to_millis,
                tx_data.loc[:, 'timestamp'].values,
                tx_data.loc[:, 'tx_kb'].values
            )
            features += [
                cls.__to_feature__(data_point, win_size, 'DATA_RX_DWT_{:02d}_COEFF'.format(idx), coeff)
                for idx, coeff in enumerate(dwt_rx)
            ]
            features += [
                cls.__to_feature__(data_point, win_size, 'DATA_TX_DWT_{:02d}_COEFF'.format(idx), coeff)
                for idx, coeff in enumerate(dwt_tx)
            ]

        return features


class ScreenProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        screen_data = data.loc[lambda x: x['type'].isin(['SCREEN_ON', 'SCREEN_OFF', 'UNLOCK']), :]
        concat_data = pd.concat([
            screen_data,
            screen_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(
            data, millis, 'timestamp'
        )

        if past_data is not None:
            features += [
                cls.__to_feature__(data_point, 0, 'SCREEN', past_data['type'], ['SCREEN_ON', 'SCREEN_OFF', 'UNLOCK'])
            ]

        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )
        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[lambda x: x['duration'] > 0, ['timestamp', 'duration', 'type', '_type']]

        for from_event, to_event in [('SCREEN_ON', 'UNLOCK'), ('SCREEN_ON', 'SCREEN_OFF'), ('UNLOCK', 'SCREEN_OFF')]:
            sub_data = window_data.loc[lambda x: (x['type'] == from_event) & (window_data['_type'] == to_event), :]
            mean_frequency = len(sub_data.index) / win_size
            duration_mean = __mean__(sub_data.loc[:, 'duration'])
            duration_std = __std__(sub_data.loc[:, 'duration'])

            features += [
                cls.__to_feature__(
                    data_point, win_size,
                    'SCREEN_{}_TO_{}_DURATION_MEAN'.format(from_event, to_event),
                    duration_mean
                ),
                cls.__to_feature__(
                    data_point, win_size,
                    'SCREEN_{}_TO_{}_DURATION_STD'.format(from_event, to_event),
                    duration_std
                ),
                cls.__to_feature__(
                    data_point, win_size,
                    'SCREEN_{}_TO_{}_FREQUENCY'.format(from_event, to_event),
                    mean_frequency
                )
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    sub_data.loc[:, 'timestamp'].values,
                    sub_data.loc[:, 'duration'].values
                )
                features += [
                    cls.__to_feature__(
                        data_point, win_size,
                        'SCREEN_{}_TO_{}_DURATION_DWT_{:02d}_COEFF'.format(from_event, to_event, idx),
                        coeff
                    ) for idx, coeff in enumerate(dwt)
                ]

        return features


class RingerModeProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        ringer_data = data.loc[
                      lambda x: x['type'].str.startswith('RINGER'), :
                      ]
        concat_data = pd.concat([
            ringer_data,
            ringer_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)
        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(
            data, millis, 'timestamp'
        )

        if past_data is not None:
            features += [
                cls.__to_feature__(
                    data_point, 0, 'RINGER', past_data['type'],
                    ['RINGER_MODE_NORMAL', 'RINGER_MODE_VIBRATE', 'RINGER_MODE_SILENT']
                )
            ]

        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )
        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[lambda x: x['duration'] > 0, ['timestamp', 'duration', 'type']]

        for mode in ['RINGER_MODE_NORMAL', 'RINGER_MODE_VIBRATE', 'RINGER_MODE_SILENT']:
            sub_data = window_data.loc[lambda x: x['type'] == mode, :]
            mean_frequency = len(sub_data.index) / win_size
            duration_mean = __mean__(sub_data.loc[:, 'duration'])
            duration_std = __std__(sub_data.loc[:, 'duration'])

            features += [
                cls.__to_feature__(data_point, win_size, '{}_DURATION_MEAN'.format(mode), duration_mean),
                cls.__to_feature__(data_point, win_size, '{}_DURATION_STD'.format(mode), duration_std),
                cls.__to_feature__(data_point, win_size, '{}_FREQUENCY'.format(mode), mean_frequency)
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    sub_data.loc[:, 'timestamp'].values,
                    sub_data.loc[:, 'duration'].values,
                )

                features += [
                    cls.__to_feature__(data_point, win_size, '{}_DURATION_DWT_{:02d}_COEFF'.format(mode, idx), coeff)
                    for idx, coeff in enumerate(dwt)
                ]

        return features


class LocationProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        begin_time = data.loc[:, 'timestamp'].min()
        end_time = data.loc[:, 'timestamp'].max()

        sampling_data = pd.DataFrame({
            'timestamp': np.arange(begin_time, end_time, 1000 * 60 * 5),  # 5-Min interval
            'latitude': np.nan,
            'longitude': np.nan,
            'is_resample': True
        })

        re_sample_data = pd.concat(
            [data, sampling_data], axis=0, sort=True
        ).sort_values('timestamp').fillna(method='ffill').loc[lambda x: ~x['is_resample'].isna(), :]
        '''
        Update macro clusters every 12 hours
        30-min stay-point will be removed 7-days after
        '''
        db_stream = DBStream(
            dim=2, radius=50 / 6371e3, decay_factor=.0012, alpha=0.3, min_weight=1.0, time_gap=144
        )

        lat_lon = np.radians(re_sample_data.loc[:, ['latitude', 'longitude']].to_numpy())
        labels = np.empty(lat_lon.shape[0], dtype='<U10')
        unlabeled_cache = []

        for i in range(lat_lon.shape[0]):
            db_stream.add_datum(lat_lon[i])

            if db_stream.is_initialized():
                if unlabeled_cache:
                    for j in range(len(unlabeled_cache)):
                        label = db_stream.get_label(unlabeled_cache[j])
                        labels[i] = label if label is not None else 'UNDEFINED'
                    unlabeled_cache.clear()

                label = db_stream.get_label(lat_lon[i])
                labels[i] = label if label is not None else 'UNDEFINED'

            else:
                unlabeled_cache.append(lat_lon[i])

        re_sample_data = pd.concat(
            [re_sample_data.reset_index(), pd.DataFrame({'label': labels})], axis=1
        )

        top_ten_labels = re_sample_data.loc[lambda x: x['label'] != 'UNDEFINED', 'label'].value_counts().index[:10]
        re_sample_data.loc[lambda x: ~x['label'].isin(top_ten_labels), 'label'] = 'UNDEFINED'

        norm_labels = {
            'label': {
                label: 'TOP-{:02d}-PLACE'.format(idx + 1) for idx, label in enumerate(top_ten_labels)
            }
        }
        re_sample_data.replace(norm_labels, inplace=True)
        diff_data = re_sample_data.loc[lambda x: x['label'] != x.shift(1)['label'], :]

        concat_data = pd.concat([
            diff_data,
            diff_data.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(
            data, millis, 'timestamp'
        )

        labels = list(data.loc[:, 'label'].unique())

        if past_data is not None:
            features += [
                cls.__to_feature__(data_point, 0, 'LOCATION', past_data['label'], labels)
            ]
        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )

        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )

        window_data = window_data.loc[lambda x: x['duration'] > 0, ['timestamp', 'label', 'duration']]

        for label in np.unique(data['label']):
            sub_data = window_data.loc[lambda x: x['label'] == label, :]
            mean_frequency = len(sub_data.index) / win_size
            duration_mean = __mean__(sub_data.loc[:, 'duration'])
            duration_std = __std__(sub_data.loc[:, 'duration'])

            features += [
                cls.__to_feature__(data_point, win_size, 'LOCATION_{}_DURATION_MEAN'.format(label), duration_mean),
                cls.__to_feature__(data_point, win_size, 'LOCATION_{}_DURATION_STD'.format(label), duration_std),
                cls.__to_feature__(data_point, win_size, 'LOCATION_{}_FREQUENCY'.format(label), mean_frequency)
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    sub_data.loc[:, 'timestamp'].values,
                    sub_data.loc[:, 'duration'].values,
                )

                features += [
                    cls.__to_feature__(
                        data_point, win_size, 'LOCATION_{}_DURATION_DWT_{:02d}_COEFF'.format(label, idx), coeff
                    ) for idx, coeff in enumerate(dwt)
                ]

        return features


class MessageProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        return []

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp'
        )

        for box, is_contact in list(product(['SENT', 'INBOX'], [True, False])):
            sub_data = window_data.loc[
                lambda x: (x['message_box'] == box) &
                          (x['contact'] != 'UNDEFINED' if is_contact else x['contact'] == 'UNDEFINED')
            ]

            contact = 'CONTACT' if is_contact else 'NON_CONTACT'
            mean_frequency = len(sub_data.index) / win_size

            features += [cls.__to_feature__(
                data_point, win_size,
                'MESSAGE_{}_{}_FREQUENCY'.format(box, contact), mean_frequency
            )]

        return features


class ActivityProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_enter = data.loc[
                     lambda x: x['transition_type'].str.startswith('ENTER'), :
                     ]
        concat_data = pd.concat([
            data_enter,
            data_enter.rename(lambda x: '_{}'.format(x), axis=1).shift(-1)
        ], axis=1)

        return concat_data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        millis = data_point * 1000

        past_data = __extract_immediate_past_data__(
            data, millis, 'timestamp'
        )

        if past_data is not None:
            features += [cls.__to_feature__(
                data_point, 0,
                'ACTIVITY_IN_PROGRESS',
                past_data['transition_type'].replace('ENTER_', ''),
                ['WALKING', 'STILL', 'IN_VEHICLE', 'ON_BICYCLE', 'RUNNING']
            )]
        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp', '_timestamp'
        )

        window_data.loc[:, 'duration'] = __duration__(
            window_data.loc[:, 'timestamp'].values, window_data.loc[:, '_timestamp'].values, from_millis, to_millis,
            win_size
        )
        window_data = window_data.loc[lambda x: x['duration'] > 0, ['timestamp', 'transition_type', 'duration']]

        for activity in ['WALKING', 'STILL', 'IN_VEHICLE', 'ON_BICYCLE', 'RUNNING']:
            sub_data = window_data.loc[lambda x: x['transition_type'] == 'ENTER_{}'.format(activity), :]
            mean_frequency = len(sub_data.index) / win_size
            duration_mean = __mean__(sub_data.loc[:, 'duration'])
            duration_std = __std__(sub_data.loc[:, 'duration'])

            features += [
                cls.__to_feature__(data_point, win_size, 'ACTIVITY_{}_DURATION_MEAN'.format(activity), duration_mean),
                cls.__to_feature__(data_point, win_size, 'ACTIVITY_{}_DURATION_STD'.format(activity), duration_std),
                cls.__to_feature__(data_point, win_size, 'ACTIVITY_{}_FREQUENCY'.format(activity), mean_frequency)
            ]

            if win_size >= 60:
                dwt = __dwt_duration__(
                    from_millis, to_millis,
                    sub_data.loc[:, 'timestamp'].values,
                    sub_data.loc[:, 'duration'].values
                )
                features += [
                    cls.__to_feature__(
                        data_point,
                        win_size,
                        'ACTIVITY_{}_DURATION_DWT_{:02d}_COEFF'.format(activity, idx),
                        coeff
                    ) for idx, coeff in enumerate(dwt)
                ]

        return features


class TimeProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        features = []
        time_obj = datetime.utcfromtimestamp(data_point) + timedelta(hours=9)

        if 9 <= time_obj.hour < 12:
            time_slot = 'MORNING'
        elif 12 <= time_obj.hour < 15:
            time_slot = 'LUNCH'
        elif 15 <= time_obj.hour < 18:
            time_slot = 'AFTERNOON'
        elif 18 <= time_obj.hour < 21:
            time_slot = 'DINNER'
        else:
            time_slot = 'NIGHT'

        if 0 <= time_obj.weekday() < 5:
            time_is_weekday = True
        else:
            time_is_weekday = False

        time_weekday = WEEKDAYS[time_obj.weekday()]
        time_norm = (time_obj.hour * 60 + time_obj.minute) / (24 * 60)

        features += [
            cls.__to_feature__(
                data_point, 0, 'TIME_{}'.format('SLOT'), time_slot,
                ['MORNING', 'LUNCH', 'AFTERNOON', 'DINNER', 'NIGHT']
            ),
            cls.__to_feature__(
                data_point, 0, 'TIME_{}'.format('IS_WEEKDAY'), time_is_weekday, [True, False]
            ),
            cls.__to_feature__(
                data_point, 0, 'TIME_{}'.format('WEEKDAY'), time_weekday, WEEKDAYS
            ),
            cls.__to_feature__(
                data_point, 0, 'TIME_{}'.format('DAY_NORM'), time_norm
            )
        ]

        return features

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        return []


class NotificationProcessor(DataProcessor):
    async def __prepare_data__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[lambda x: ~x['package_name'].isin(APP_EXPERIMENT), :]

    @classmethod
    async def __generate_current_feature__(cls, data_point: int, data: pd.DataFrame) -> List[Feature]:
        return []

    @classmethod
    async def __generate_window_feature__(cls, data_point: int, win_size: int, data: pd.DataFrame) -> List[Feature]:
        features = []

        from_millis = (data_point - win_size) * 1000
        to_millis = data_point * 1000

        window_data = __extract_window_data__(
            data, from_millis, to_millis, 'timestamp'
        )
        window_data = window_data.loc[:, ['timestamp', 'package_name']]

        mean_frequency = len(window_data.index) / win_size
        features += [
            cls.__to_feature__(data_point, win_size, 'NOTI_TOTAL_FREQUENCY_PER_SECOND', mean_frequency)
        ]

        for category, apps in APP_CATEGORIES.items():
            category_data = window_data.loc[lambda x: x['package_name'].isin(apps), :]
            mean_frequency_by_category = len(category_data.index) / win_size

            features += [
                cls.__to_feature__(
                    data_point, win_size, 'NOTI_{}_FREQUENCY_PER_SECOND'.format(category), mean_frequency_by_category
                )
            ]

        return features
