$(document).ready(function() {
    $('.page__content img').wrap( function(){

        // 2.2. magnific-popup 옵션 설정.
        $(this).magnificPopup({
            type: 'image',
            enableEscapeKey: true,
            closeOnBgClick: true,
            closeOnContentClick: true,
            showCloseBtn: true,
            items: {
              src: $(this).attr('src')
            },
        });

        // 2.3. p 태그 높이를 내부 컨텐츠 높이에 자동으로 맞추기.
        $(this).parent('p').css('overflow', 'auto');
        });
});